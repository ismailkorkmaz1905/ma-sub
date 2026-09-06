import copy
import math
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest


def _amount(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise IntegrityError(f"qualification {label} is invalid") from None
    if not math.isfinite(result) or result <= 0:
        raise IntegrityError(f"qualification {label} is invalid")
    return result


def _nonnegative_amount(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise IntegrityError(f"qualification {label} is invalid") from None
    if not math.isfinite(result) or result < 0:
        raise IntegrityError(f"qualification {label} is invalid")
    return result


def _matches(pod, payload, old_ids, requested_after):
    if not _owned(pod, payload, old_ids, requested_after):
        return False
    machine = pod.get("machine") or {}
    return (pod.get("networkVolumeId") == payload["networkVolumeId"]
            and pod.get("imageName") == payload["imageName"]
            and machine.get("dataCenterId", pod.get("dataCenterId")) == payload["dataCenterId"]
            and machine.get("gpuTypeId", pod.get("gpuTypeId")) == payload["gpuTypeId"])


def _owned(pod, payload, old_ids, requested_after):
    return (isinstance(pod, dict) and pod.get("id") not in old_ids
            and pod.get("name") == payload["name"]
            and _timestamp(pod.get("createdAt")) >= int(_timestamp(requested_after)) - 1)


def _timestamp(value):
    text = str(value)
    try:
        if text.endswith(" UTC"):
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f %z UTC").timestamp()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return float("-inf")
        return parsed.timestamp()
    except (TypeError, ValueError):
        return float("-inf")


def _reconcile(provider, payload, old_ids, requested_after, deadline, *, clock, sleep):
    while clock() < deadline:
        matches = [pod for pod in provider.list_pods(min(10, deadline - clock()))
                   if _owned(pod, payload, old_ids, requested_after)]
        if matches:
            return matches
        sleep(min(1, max(0, deadline - clock())))
    return []


def run_disposable_qualification(provider, base_payload, worker, output_dir, *,
                                  maximum_rate_usd_per_hour, lease_seconds=550,
                                  minimum_balance_usd=1.10,
                                  protected_pod_id="781ct55zv4gkle",
                                  shutdown_reserve_seconds=120,
                                  reconciliation_seconds=30, clock=time.monotonic,
                                  wall_clock=time.time, sleep=time.sleep, nonce=None):
    if (not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool)
            or not 1 <= lease_seconds <= 550):
        raise ValueError("qualification lease must be within 1..550 seconds")
    maximum_rate = _amount(maximum_rate_usd_per_hour, "rate cap")
    reserve = _amount(minimum_balance_usd, "balance reserve")
    if not isinstance(shutdown_reserve_seconds, int) or not 1 <= shutdown_reserve_seconds < lease_seconds:
        raise ValueError("qualification shutdown reserve is invalid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    old_pods = provider.list_pods(10)
    old_ids = sorted(pod["id"] for pod in old_pods)
    if protected_pod_id not in old_ids:
        raise IntegrityError("protected qualification Pod is missing")
    if next(pod for pod in old_pods if pod["id"] == protected_pod_id).get("desiredStatus") != "EXITED":
        raise IntegrityError("protected qualification Pod must remain EXITED")
    account = provider.account(10)
    balance = _amount(account.get("clientBalance"), "balance")
    current_rate = _nonnegative_amount(account.get("currentSpendPerHr"), "account rate")
    if account.get("isAutoPayEnabled") is not False:
        raise IntegrityError("qualification requires disabled auto-pay")
    volume = provider.get_volume(base_payload.get("networkVolumeId"), 10)
    if (volume.get("id") != "xgogcmey5o"
            or volume.get("dataCenterId") != base_payload.get("dataCenterId")):
        raise IntegrityError("qualification volume identity is invalid")
    maximum_charge = (current_rate + maximum_rate) * lease_seconds / 3600
    if balance - maximum_charge < reserve:
        raise IntegrityError("qualification cost would breach balance reserve")
    token = nonce or secrets.token_hex(8)
    if not re.fullmatch(r"[a-f0-9]{16}", token):
        raise ValueError("qualification nonce must be 16 lowercase hex characters")
    requested_epoch = wall_clock()
    requested_after = datetime.fromtimestamp(requested_epoch, timezone.utc).isoformat()
    terminate_after = datetime.fromtimestamp(requested_epoch + lease_seconds,
                                              timezone.utc).isoformat()
    payload = copy.deepcopy(base_payload)
    payload.update(name=f"mas-ep11-qualification-{token}", terminateAfter=terminate_after)
    required = {"networkVolumeId", "imageName", "dataCenterId", "gpuTypeId"}
    if not required <= payload.keys() or payload["networkVolumeId"] != "xgogcmey5o":
        raise IntegrityError("qualification payload identity is incomplete")
    state = {"format": "mas-disposable-qualification-state-1", "payload": payload,
             "payload_sha256": digest(payload), "old_pod_ids": old_ids,
             "requested_at_utc": requested_after, "lease_seconds": lease_seconds,
             "maximum_rate_usd_per_hour": maximum_rate}
    state.update(balance_usd=balance, current_rate_usd_per_hour=current_rate,
                 minimum_balance_usd=reserve, maximum_charge_usd=maximum_charge,
                 protected_pod_id=protected_pod_id)
    atomic_json(output_dir / "create-state.json", {"data": state, "sha256": digest(state)})
    deadline = clock() + lease_seconds
    owned = []
    outcome = None
    completed = False
    failure_type = None
    try:
        try:
            remaining = deadline - clock()
            if remaining <= shutdown_reserve_seconds:
                raise IntegrityError("qualification lease expired before create")
            response = provider.create(copy.deepcopy(payload), min(20, remaining))
            candidates = [response] if _owned(response, payload, set(old_ids), requested_after) else []
        except Exception as exc:
            failure_type = type(exc).__name__
            candidates = []
        if not candidates:
            candidates = _reconcile(provider, payload, set(old_ids), requested_after,
                                    min(deadline, clock() + reconciliation_seconds),
                                    clock=clock, sleep=sleep)
        owned = candidates
        if len(candidates) != 1:
            raise IntegrityError("qualification create could not resolve one exact new Pod")
        pod = candidates[0]
        if not _matches(pod, payload, set(old_ids), requested_after):
            raise IntegrityError("qualification created Pod does not match requested configuration")
        if pod["id"] in old_ids:
            raise IntegrityError("qualification create returned an old Pod")
        if _amount(pod.get("costPerHr"), "Pod rate") > maximum_rate:
            raise IntegrityError("qualification Pod rate exceeds cap")
        work_seconds = deadline - clock() - shutdown_reserve_seconds
        if work_seconds <= 0:
            raise IntegrityError("qualification lease has no worker budget")
        outcome = worker(pod, work_seconds)
        completed = True
        return outcome
    finally:
        shutdown = []
        for pod in owned:
            if not _owned(pod, payload, set(old_ids), requested_after):
                continue
            error = None
            try:
                provider.terminate(pod["id"], min(15, max(0.001, deadline - clock())))
                remaining = max(0.001, deadline - clock())
                provider.wait_absent(pod["id"], remaining)
                status = "ABSENT"
            except Exception as exc:
                status, error = "UNVERIFIED", type(exc).__name__
            shutdown.append({"pod_id": pod["id"], "status": status, "error_type": error})
        receipt = {"format": "mas-disposable-qualification-receipt-1",
                   "state_sha256": digest(state), "owned_pods": shutdown,
                   "provider_terminate_after": terminate_after,
                   "provider_backstop_verified": False,
                   "completed": completed, "provider_error_type": failure_type}
        atomic_json(output_dir / "shutdown-receipt.json",
                    {"data": receipt, "sha256": digest(receipt)})
        if owned and any(item["status"] != "ABSENT" for item in shutdown):
            raise IntegrityError("qualification Pod termination was not verified")
