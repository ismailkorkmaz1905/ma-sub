import copy
import json
import math
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from .reliability import IntegrityError, atomic_json, digest
from .subtitle.pilot_runner import PilotProviderError
from .subtitle.qualification_provider import QualificationProvider


DEFAULT_GPU_TYPE_IDS = ("NVIDIA L4", "NVIDIA RTX 4000 Ada Generation")
STORAGE_PRICING_URL = "https://docs.runpod.io/pods/storage/types"


def load_storage_quote(path):
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        quote = saved["data"]
    except (OSError, ValueError, KeyError, TypeError):
        raise IntegrityError("MAS_RUNPOD_STORAGE_QUOTE JSON is missing or invalid") from None
    required = {"observed_at_utc", "source_url", "network_volume_usd_per_gb_month",
                "container_storage_usd_per_gb_month"}
    if (not isinstance(quote, dict) or saved.get("sha256") != digest(quote)
            or set(quote) != required or quote.get("source_url") != STORAGE_PRICING_URL):
        raise IntegrityError("storage quote identity or SHA-256 is invalid")
    observed = _timestamp(quote.get("observed_at_utc"))
    if not math.isfinite(observed) or not -5 <= time.time() - observed <= 86400:
        raise IntegrityError("storage quote must be observed within 24 hours")
    _money(quote.get("network_volume_usd_per_gb_month"), "network volume quote", positive=True)
    _money(quote.get("container_storage_usd_per_gb_month"), "container storage quote", positive=True)
    return {**quote, "sha256": digest(quote)}


class CapacityProvider(QualificationProvider):
    def offers(self, gpu_type_ids, data_center_id, timeout):
        if data_center_id != "EU-RO-1":
            raise IntegrityError("capacity offers are restricted to EU-RO-1")
        query = (
            'query { gpuTypes { id lowestPrice(input: {gpuCount: 1, secureCloud: true, '
            'dataCenterId: "EU-RO-1", supportPublicIp: true}) { stockStatus '
            'uninterruptablePrice availableGpuCounts } } }')
        request = self._graphql_request({"query": query})
        result = self._json(request, timeout)
        if result.get("errors") or not isinstance(result.get("data", {}).get("gpuTypes"), list):
            raise IntegrityError("RunPod GPU offer query failed")
        allowed = set(gpu_type_ids)
        offers = []
        for gpu in result["data"]["gpuTypes"]:
            price = gpu.get("lowestPrice") or {}
            if gpu.get("id") in allowed:
                offers.append({"gpuTypeId": gpu["id"], "dataCenterId": data_center_id,
                               "costPerHr": price.get("uninterruptablePrice"),
                               "available": price.get("stockStatus") in ("Low", "Medium", "High")})
        return offers

    def _graphql_request(self, payload):
        import urllib.parse
        import urllib.request
        url = "https://api.runpod.io/graphql?api_key=" + urllib.parse.quote(self.api_key, safe="")
        return urllib.request.Request(url, data=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})

    def create(self, payload, timeout):
        deadline = time.monotonic() + timeout
        query = "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id } }"
        result = self._json(self._graphql_request(
            {"query": query, "variables": {"input": payload}}), timeout)
        errors = result.get("errors")
        if errors:
            messages = [str(item.get("message", "")) for item in errors if isinstance(item, dict)]
            if messages and all("not enough free gpu" in message.lower() for message in messages):
                raise PilotProviderError({"category": "capacity", "method": "POST"})
            raise IntegrityError("capacity create GraphQL errors; reconcile without retry")
        pod_id = result.get("data", {}).get("podFindAndDeployOnDemand", {}).get("id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id or ""):
            raise IntegrityError("capacity create response has no valid Pod ID")
        return self.get_pod(pod_id, max(0.001, deadline - time.monotonic()))


class CapacityReadinessError(RuntimeError):
    pass


def _record_create_error(attempt, exc):
    attempt["create_error_type"] = type(exc).__name__
    if isinstance(exc, PilotProviderError):
        for key in ("category", "http_status", "method"):
            value = exc.safe_details.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                attempt["create_provider_" + key] = value


class CapacityPlan:
    def __init__(self, *, episode, maximum_rate_usd_per_hour,
                 gpu_type_ids=DEFAULT_GPU_TYPE_IDS,
                 total_seconds, startup_seconds=300, shutdown_seconds=120,
                 reserve_usd=0.0, billing_margin_usd=0.0,
                 storage_quote=None,
                 protected_pod_id="781ct55zv4gkle",
                 network_volume_id="xgogcmey5o", data_center_id="EU-RO-1"):
        if type(episode) is not int or episode < 1:
            raise ValueError("episode must be a positive integer")
        if (not isinstance(gpu_type_ids, (list, tuple)) or not gpu_type_ids
                or len(gpu_type_ids) > 8 or len(set(gpu_type_ids)) != len(gpu_type_ids)
                or any(not re.fullmatch(r"[A-Za-z0-9 ._-]+", value or "")
                       for value in gpu_type_ids)):
            raise ValueError("GPU types must be 1-8 unique provider IDs")
        if (not isinstance(maximum_rate_usd_per_hour, (int, float))
                or isinstance(maximum_rate_usd_per_hour, bool)
                or not math.isfinite(maximum_rate_usd_per_hour)
                or maximum_rate_usd_per_hour <= 0):
            raise ValueError("capacity maximum rate must be finite and positive")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0
               for value in (reserve_usd, billing_margin_usd)):
            raise ValueError("capacity reserves must be finite and nonnegative")
        if (not isinstance(storage_quote, dict)
                or storage_quote.get("source_url") != STORAGE_PRICING_URL
                or storage_quote.get("sha256") != digest(
                    {key: value for key, value in storage_quote.items() if key != "sha256"})
                or not -5 <= time.time() - _timestamp(storage_quote.get("observed_at_utc")) <= 86400):
            raise ValueError("a valid storage quote observed within 24 hours is required")
        if (type(total_seconds) is not int or type(startup_seconds) is not int
                or type(shutdown_seconds) is not int
                or not total_seconds > startup_seconds + shutdown_seconds > 0):
            raise ValueError("capacity deadlines are invalid")
        if data_center_id != "EU-RO-1" or network_volume_id != "xgogcmey5o":
            raise ValueError("capacity storage scope is fixed to the retained EU-RO-1 volume")
        self.episode = episode
        self.gpu_type_ids = tuple(gpu_type_ids)
        self.maximum_rate_usd_per_hour = float(maximum_rate_usd_per_hour)
        self.total_seconds = total_seconds
        self.startup_seconds = startup_seconds
        self.shutdown_seconds = shutdown_seconds
        self.reserve_usd = float(reserve_usd)
        self.billing_margin_usd = float(billing_margin_usd)
        self.container_storage_usd_per_gb_month = _money(
            storage_quote.get("container_storage_usd_per_gb_month"),
            "container storage quote", positive=True)
        self.network_volume_usd_per_gb_month = _money(
            storage_quote.get("network_volume_usd_per_gb_month"),
            "network volume quote", positive=True)
        self.storage_quote = dict(storage_quote)
        self.protected_pod_id = protected_pod_id
        self.network_volume_id = network_volume_id
        self.data_center_id = data_center_id


class CapacityLease:
    def __init__(self, provider, base_payload, audit_dir, plan, *,
                 clock=time.monotonic, wall_clock=time.time, sleep=time.sleep,
                 nonce=None, ready=None, resume=False,
                 resume_reconciliation_seconds=30):
        self.provider = provider
        self.base_payload = copy.deepcopy(base_payload)
        self.audit_dir = Path(audit_dir)
        self.plan = plan
        self.clock = clock
        self.wall_clock = wall_clock
        self.sleep = sleep
        self.nonce = nonce or secrets.token_hex(8)
        if not re.fullmatch(r"[a-f0-9]{16}", self.nonce):
            raise ValueError("capacity nonce must be 16 lowercase hex characters")
        self.ready = ready
        self.resume = resume
        self.was_resumed = False
        if (not isinstance(resume_reconciliation_seconds, (int, float))
                or isinstance(resume_reconciliation_seconds, bool)
                or not math.isfinite(resume_reconciliation_seconds)
                or resume_reconciliation_seconds < 0):
            raise ValueError("resume reconciliation time must be finite and nonnegative")
        self.resume_reconciliation_seconds = float(resume_reconciliation_seconds)
        self.pod = None
        self.work_budget_seconds = None
        self._state = None
        self._old_ids = set()
        self._deadline = None
        self._startup_deadline = None
        self._requested_epoch = None

    def __enter__(self):
        try:
            self.acquire()
            return self
        except BaseException:
            if self._state is not None:
                self.cleanup()
            raise

    def __exit__(self, exc_type, exc, traceback):
        self.cleanup()

    def remaining_work_seconds(self):
        if self.pod is None or self._deadline is None:
            raise IntegrityError("capacity lease is not acquired")
        return max(0, self._deadline - self.clock() - self.plan.shutdown_seconds)

    def _left(self, cap):
        left = self._deadline - self.clock()
        if left <= 0:
            raise TimeoutError("capacity lease deadline expired")
        return min(cap, left)

    def _save(self):
        atomic_json(self.audit_dir / "capacity-state.json",
                    {"data": self._state, "sha256": digest(self._state)})

    def _owned(self, pod, name):
        return (isinstance(pod, dict)
                and re.fullmatch(r"[A-Za-z0-9_-]+", pod.get("id") or "") is not None
                and pod.get("id") not in self._old_ids and pod.get("name") == name
                and _timestamp(pod.get("createdAt")) >= self._requested_epoch - 1)

    def _reconcile(self, name):
        end = min(self._startup_deadline, self._deadline - self.plan.shutdown_seconds,
                  self.clock() + self.plan.startup_seconds)
        while self.clock() < end:
            matches = [pod for pod in self.provider.list_pods(self._left(10))
                       if self._owned(pod, name)]
            if matches:
                return matches
            self.sleep(min(1, max(0, end - self.clock())))
        return []

    def acquire(self):
        if self.resume:
            return self._resume()
        if self.audit_dir.exists():
            self._archive_released_state()
        else:
            self.audit_dir.mkdir(parents=True)
        self._deadline = self.clock() + self.plan.total_seconds
        self._startup_deadline = self.clock() + self.plan.startup_seconds
        old_pods = self.provider.list_pods(self._left(10))
        self._old_ids = {pod.get("id") for pod in old_pods}
        protected = [pod for pod in old_pods if pod.get("id") == self.plan.protected_pod_id]
        if len(protected) != 1 or protected[0].get("desiredStatus") != "EXITED":
            raise IntegrityError("protected retained Pod must exist and remain EXITED")
        if any(pod.get("desiredStatus") != "EXITED" for pod in old_pods):
            raise IntegrityError("unrelated active Pod prevents capacity acquisition")
        volume = self.provider.get_volume(self.plan.network_volume_id, self._left(10))
        if (volume.get("id") != self.plan.network_volume_id
                or volume.get("dataCenterId") != self.plan.data_center_id):
            raise IntegrityError("retained network volume identity changed")
        volume_gb = _money(volume.get("size"), "network volume size", positive=True)
        account = self.provider.account(self._left(10))
        balance = _money(account.get("clientBalance"), "account balance", positive=True)
        account_rate = _money(account.get("currentSpendPerHr"), "account rate")
        minimum_balance = self.plan.reserve_usd + self.plan.billing_margin_usd
        if balance < minimum_balance:
            raise IntegrityError("live account balance is below the protected reserve")
        if account.get("isAutoPayEnabled") is not False:
            raise IntegrityError("capacity acquisition requires disabled auto-pay")
        self._requested_epoch = self.wall_clock()
        requested_at = datetime.fromtimestamp(self._requested_epoch, timezone.utc).isoformat()
        self._state = {
            "format": "mas-capacity-lease-state-1", "episode": self.plan.episode,
            "requested_at_utc": requested_at, "protected_pod_id": self.plan.protected_pod_id,
            "network_volume_id": self.plan.network_volume_id,
            "network_volume_size_gb": volume_gb,
            "data_center_id": self.plan.data_center_id, "old_pod_ids": sorted(self._old_ids),
            "balance_usd": balance, "minimum_balance_usd": minimum_balance,
            "account_rate_usd_per_hour": account_rate,
            "storage_quote_sha256": self.plan.storage_quote["sha256"],
            "storage_quote_observed_at_utc": self.plan.storage_quote["observed_at_utc"],
            "storage_quote_source_url": self.plan.storage_quote["source_url"],
            "attempts": [], "owned_pod_ids": [], "status": "WAITING_FOR_CAPACITY",
        }
        self._save()
        while True:
            try:
                offers = _ordered_offers(
                    self.provider.offers(
                        self.plan.gpu_type_ids,
                        self.plan.data_center_id,
                        self._left(20),
                    ),
                    self.plan,
                )
                break
            except IntegrityError as exc:
                if str(exc) != "no fresh allowed EU-RO-1 GPU offers are available":
                    raise
                left = min(
                    self._startup_deadline,
                    self._deadline - self.plan.shutdown_seconds,
                ) - self.clock()
                if left <= 1:
                    self._state["status"] = "NO_CAPACITY"
                    self._save()
                    raise
                self.sleep(min(1, left))
        self._state["status"] = "READY_TO_CREATE"
        self._save()
        for index, offer in enumerate(offers):
            inventory = self.provider.list_pods(self._left(10))
            if any(pod.get("desiredStatus") != "EXITED" for pod in inventory):
                raise IntegrityError("active Pod appeared before capacity create")
            account = self.provider.account(self._left(10))
            balance = _money(account.get("clientBalance"), "account balance", positive=True)
            account_rate = _money(account.get("currentSpendPerHr"), "account rate")
            if account.get("isAutoPayEnabled") is not False:
                raise IntegrityError("capacity acquisition requires disabled auto-pay")
            disk_gb = _money(self.base_payload.get("containerDiskInGb", 0),
                             "container disk size")
            storage_rate = disk_gb * self.plan.container_storage_usd_per_gb_month / (30 * 24)
            storage_rate += volume_gb * self.plan.network_volume_usd_per_gb_month / (30 * 24)
            affordable_seconds = math.floor(
                (balance - minimum_balance) * 3600 /
                (account_rate + self.plan.maximum_rate_usd_per_hour + storage_rate))
            lease_seconds = min(self.plan.total_seconds, affordable_seconds)
            if lease_seconds <= self.plan.shutdown_seconds or self.clock() >= self._startup_deadline:
                continue
            terminate_after = datetime.fromtimestamp(
                self.wall_clock() + lease_seconds, timezone.utc).isoformat()
            offer_deadline = self.clock() + lease_seconds
            name = f"mas-ep{self.plan.episode}-production-{self.nonce}-{index}"
            payload = copy.deepcopy(self.base_payload)
            payload.update(name=name, gpuTypeId=offer["gpu_type_id"],
                           networkVolumeId=self.plan.network_volume_id,
                           dataCenterId=self.plan.data_center_id,
                           terminateAfter=terminate_after)
            attempt = {"name": name, "gpu_type_id": offer["gpu_type_id"],
                       "quoted_rate_usd_per_hour": offer["rate"], "status": "CREATE_REQUESTED",
                       "payload_sha256": digest(payload),
                       "provider_terminate_after": terminate_after,
                       "balance_usd_before_create": balance}
            self._state["attempts"].append(attempt)
            self._state["status"] = "CREATE_REQUESTED"
            self._save()
            try:
                created = self.provider.create(payload, self._left(20))
            except PilotProviderError as exc:
                _record_create_error(attempt, exc)
                if exc.safe_details.get("category") == "capacity":
                    attempt["status"] = "UNAMBIGUOUS_NO_CREATE"
                    self._save()
                    continue
                attempt["status"] = "AMBIGUOUS_RECONCILING"
                self._save()
                matches = self._reconcile(name)
            except BaseException as exc:
                _record_create_error(attempt, exc)
                attempt["status"] = "AMBIGUOUS_RECONCILING"
                self._save()
                matches = self._reconcile(name)
                self._state["owned_pod_ids"] = [pod["id"] for pod in matches]
                self._save()
                if not isinstance(exc, Exception):
                    raise
            else:
                matches = [created] if self._owned(created, name) else self._reconcile(name)
            if len(matches) != 1:
                self._state["owned_pod_ids"] = [pod["id"] for pod in matches]
                attempt["status"] = "AMBIGUOUS_UNRESOLVED"
                self._save()
                raise IntegrityError("capacity create did not reconcile to one owned Pod")
            self.pod = matches[0]
            self._state["owned_pod_ids"] = [self.pod.get("id")]
            attempt["status"] = "OWNED"
            self._state["status"] = "OWNED"
            self._save()
            try:
                self._validate_pod(payload, offer)
                if self.ready is not None:
                    readiness_left = self._startup_deadline - self.clock()
                    if readiness_left <= 0:
                        raise TimeoutError("shared capacity startup deadline expired")
                    self.ready(self.pod, readiness_left)
            except CapacityReadinessError as exc:
                attempt["status"] = "READINESS_FAILED"
                attempt["readiness_error_type"] = type(exc).__name__
                self._save()
                attempt["shutdown"] = self._cleanup_pods([self.pod])
                if any(item["status"] != "ABSENT" for item in attempt["shutdown"]):
                    self._save()
                    raise IntegrityError(
                        "owned temporary Pod termination was not externally verified ABSENT")
                self.pod = None
                self._state["owned_pod_ids"] = []
                self._save()
                continue
            self._deadline = min(self._deadline, offer_deadline)
            self.work_budget_seconds = self.remaining_work_seconds()
            if self.work_budget_seconds <= 0:
                raise IntegrityError("capacity lease has no effective work budget")
            return self
        self._state["status"] = "NO_CAPACITY"
        self._save()
        raise IntegrityError("no affordable allowed GPU capacity was created")

    def _archive_released_state(self):
        state_path = self.audit_dir / "capacity-state.json"
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            state = saved["data"]
        except (OSError, ValueError, KeyError, TypeError):
            raise IntegrityError("existing capacity audit is not reusable") from None
        if saved.get("sha256") != digest(state) or state.get("status") != "RELEASED":
            raise IntegrityError("existing capacity audit is unresolved; resume it")
        suffix = digest(state)[:12]
        state_path.replace(self.audit_dir / f"capacity-state.released-{suffix}.json")
        shutdown = self.audit_dir / "capacity-shutdown.json"
        if shutdown.exists():
            shutdown.replace(self.audit_dir / f"capacity-shutdown.released-{suffix}.json")

    def _resume(self):
        state_path = self.audit_dir / "capacity-state.json"
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            state = saved["data"]
        except (OSError, ValueError, KeyError, TypeError):
            raise IntegrityError("capacity resume state is missing or invalid") from None
        if not isinstance(state, dict) or saved.get("sha256") != digest(state):
            raise IntegrityError("capacity resume state checksum mismatch")
        if (state.get("format") != "mas-capacity-lease-state-1"
                or state.get("episode") != self.plan.episode
                or state.get("protected_pod_id") != self.plan.protected_pod_id
                or state.get("network_volume_id") != self.plan.network_volume_id
                or state.get("data_center_id") != self.plan.data_center_id):
            raise IntegrityError("capacity resume identity mismatch")
        if state.get("status") == "RELEASED":
            raise IntegrityError("released capacity state cannot be resumed")
        self._state = state
        self._deadline = self.clock() + self.plan.shutdown_seconds
        self._old_ids = set(state.get("old_pod_ids") or [])
        self._requested_epoch = _timestamp(state.get("requested_at_utc"))
        if not math.isfinite(self._requested_epoch):
            raise IntegrityError("capacity resume request timestamp is invalid")
        attempts = state.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise IntegrityError("capacity resume ownership journal is incomplete")
        names = {attempt.get("name") for attempt in attempts if isinstance(attempt, dict)}
        if None in names or len(names) != len(attempts):
            raise IntegrityError("capacity resume ownership names are invalid")
        inventory = self.provider.list_pods(10)
        def owned_candidates(pods):
            return [pod for pod in pods
                    if re.fullmatch(r"[A-Za-z0-9_-]+", pod.get("id") or "") is not None
                    and pod.get("id") not in self._old_ids and pod.get("name") in names
                    and _timestamp(pod.get("createdAt")) >= self._requested_epoch - 1]
        candidates = owned_candidates(inventory)
        if (not candidates and state.get("status") in
                ("CREATE_REQUESTED", "AMBIGUOUS_RECONCILING", "AMBIGUOUS_UNRESOLVED")):
            reconcile_end = self.clock() + self.resume_reconciliation_seconds
            while self.clock() < reconcile_end:
                inventory = self.provider.list_pods(min(10, max(0.001, reconcile_end - self.clock())))
                candidates = owned_candidates(inventory)
                if candidates:
                    break
                self.sleep(min(1, max(0, reconcile_end - self.clock())))
        active = [pod for pod in inventory if pod.get("desiredStatus") != "EXITED"]
        if any(pod not in candidates for pod in active):
            raise IntegrityError("unrelated active Pod prevents capacity resume")
        recorded_ids = state.get("owned_pod_ids") or []
        if recorded_ids and not candidates and all(
                pod.get("id") not in set(recorded_ids) for pod in inventory):
            state["shutdown"] = [{"pod_id": pod_id, "status": "ABSENT", "error_type": None}
                                 for pod_id in recorded_ids]
            state["status"] = "RELEASED"
            self._save()
            body = {"state_sha256": digest(state), "owned_pods": state["shutdown"]}
            atomic_json(self.audit_dir / "capacity-shutdown.json",
                        {"data": body, "sha256": digest(body)})
            raise IntegrityError("recorded capacity Pod is already externally ABSENT")
        if recorded_ids and ({pod.get("id") for pod in candidates} != set(recorded_ids)):
            raise IntegrityError("capacity resume inventory does not match recorded ownership")
        if len(candidates) != 1 or candidates[0].get("desiredStatus") != "RUNNING":
            raise IntegrityError("capacity resume did not resolve one active owned Pod")
        self.pod = candidates[0]
        attempt = next((item for item in attempts if item.get("name") == self.pod.get("name")), None)
        if attempt is None or attempt.get("gpu_type_id") not in self.plan.gpu_type_ids:
            raise IntegrityError("capacity resume GPU ownership is invalid")
        terminate_epoch = _timestamp(attempt.get("provider_terminate_after"))
        provider_seconds = terminate_epoch - self.wall_clock()
        if provider_seconds <= self.plan.shutdown_seconds:
            raise IntegrityError("capacity resume has no time before provider termination")
        account = self.provider.account(10)
        balance = _money(account.get("clientBalance"), "account balance", positive=True)
        account_rate = _money(account.get("currentSpendPerHr"), "account rate")
        if (account.get("isAutoPayEnabled") is not False
                or balance < self.plan.reserve_usd + self.plan.billing_margin_usd):
            raise IntegrityError("capacity resume account safety check failed")
        disk_gb = _money(self.base_payload.get("containerDiskInGb", 0), "container disk size")
        storage_rate = disk_gb * self.plan.container_storage_usd_per_gb_month / (30 * 24)
        storage_rate += _money(state.get("network_volume_size_gb"), "network volume size",
                               positive=True) * self.plan.network_volume_usd_per_gb_month / (30 * 24)
        affordable_seconds = math.floor(
            (balance - self.plan.reserve_usd - self.plan.billing_margin_usd) * 3600 /
            (account_rate + self.plan.maximum_rate_usd_per_hour + storage_rate))
        self._deadline = self.clock() + min(provider_seconds, affordable_seconds)
        self._startup_deadline = self.clock()
        self._validate_pod({"name": self.pod["name"]},
                           {"gpu_type_id": attempt["gpu_type_id"]})
        self.work_budget_seconds = self.remaining_work_seconds()
        if self.work_budget_seconds <= 0:
            raise IntegrityError("capacity resume has no affordable work budget")
        if self.ready is not None:
            self.ready(self.pod, min(self.plan.startup_seconds, self.work_budget_seconds))
        state["status"] = "RESUMED"
        state["resume_balance_usd"] = balance
        state["resume_storage_quote_sha256"] = self.plan.storage_quote["sha256"]
        self.was_resumed = True
        self._save()
        return self

    def _validate_pod(self, payload, offer):
        machine = self.pod.get("machine") or {}
        rate = _money(self.pod.get("costPerHr", self.pod.get("adjustedCostPerHr")),
                      "created Pod rate", positive=True)
        if (self.pod.get("networkVolumeId") != self.plan.network_volume_id
                or self.pod.get("name") != payload["name"]
                or machine.get("dataCenterId", self.pod.get("dataCenterId")) != self.plan.data_center_id
                or machine.get("gpuTypeId", self.pod.get("gpuTypeId")) != offer["gpu_type_id"]
                or rate > self.plan.maximum_rate_usd_per_hour):
            raise IntegrityError("created Pod does not match its owned capacity request")

    def cleanup(self):
        if self._state is None:
            return
        names = {attempt["name"] for attempt in self._state["attempts"]}
        owned = []
        inventory_ok = False
        try:
            inventory = self.provider.list_pods(self._left(10))
            inventory_ok = True
            owned = [pod for pod in inventory if any(self._owned(pod, name) for name in names)]
        except Exception:
            if self.pod is not None:
                owned = [self.pod]
        results = self._cleanup_pods(owned)
        unresolved = self._state.get("status") in (
            "CREATE_REQUESTED", "AMBIGUOUS_RECONCILING", "AMBIGUOUS_UNRESOLVED")
        if not owned and unresolved:
            self._state["shutdown"] = []
            self._save()
            raise IntegrityError("ambiguous capacity ownership remains unresolved")
        recorded_ids = set(self._state.get("owned_pod_ids") or [])
        recorded_ids_absent = (inventory_ok and recorded_ids and all(
            pod.get("id") not in recorded_ids for pod in inventory))
        if not owned and recorded_ids_absent:
            results = [{"pod_id": pod_id, "status": "ABSENT", "error_type": None}
                       for pod_id in sorted(recorded_ids)]
        self._state["shutdown"] = results
        self._state["status"] = "RELEASED" if results and all(
            item["status"] == "ABSENT" for item in results) else (
                "NO_CAPACITY" if self._state.get("status") == "NO_CAPACITY" and inventory_ok
                else "UNVERIFIED")
        self._save()
        body = {"state_sha256": digest(self._state), "owned_pods": results}
        atomic_json(self.audit_dir / "capacity-shutdown.json",
                    {"data": body, "sha256": digest(body)})
        if any(item["status"] != "ABSENT" for item in results):
            raise IntegrityError("owned temporary Pod termination was not externally verified ABSENT")
        if self._state["status"] == "UNVERIFIED":
            raise IntegrityError("owned temporary Pod absence could not be externally verified")

    def _cleanup_pods(self, owned):
        results = []
        for pod in owned:
            pod_id = pod.get("id")
            status = "UNVERIFIED"
            error_type = None
            try:
                self.provider.terminate(pod_id, self._left(15))
                self.provider.wait_absent(pod_id, self._left(self.plan.shutdown_seconds))
                status = "ABSENT"
            except Exception as exc:
                error_type = type(exc).__name__
            results.append({"pod_id": pod_id, "status": status, "error_type": error_type})
        return results


def _money(value, label, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise IntegrityError(f"{label} is invalid") from None
    if not math.isfinite(result) or result < 0 or positive and result <= 0:
        raise IntegrityError(f"{label} is invalid")
    return result


def _timestamp(value):
    try:
        text = str(value)
        if text.endswith(" UTC"):
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f %z UTC").timestamp()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def _ordered_offers(values, plan):
    if not isinstance(values, list):
        raise IntegrityError("capacity offers are not a list")
    rank = {gpu_type: index for index, gpu_type in enumerate(plan.gpu_type_ids)}
    offers = []
    for value in values:
        if not isinstance(value, dict) or value.get("gpuTypeId") not in rank:
            continue
        if value.get("dataCenterId") != plan.data_center_id or value.get("available") is not True:
            continue
        rate = _money(value.get("costPerHr"), "capacity offer rate", positive=True)
        if rate <= plan.maximum_rate_usd_per_hour:
            offers.append({"gpu_type_id": value["gpuTypeId"], "rate": rate})
    offers.sort(key=lambda item: (rank[item["gpu_type_id"]], item["rate"]))
    if not offers:
        raise IntegrityError("no fresh allowed EU-RO-1 GPU offers are available")
    return offers
