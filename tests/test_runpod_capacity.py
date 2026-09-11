import json
from datetime import datetime, timezone

import pytest

from mas.reliability import IntegrityError, digest
from mas.runpod_capacity import (CapacityLease, CapacityPlan, CapacityProvider,
                                 CapacityReadinessError, STORAGE_PRICING_URL,
                                 load_storage_quote)
from mas.subtitle.pilot_runner import PilotProviderError


class Provider:
    def __init__(self):
        self.actions = []
        self.created = []
        self.pods = [{"id": "781ct55zv4gkle", "desiredStatus": "EXITED"}]
        self.create_results = []

    def list_pods(self, timeout):
        return list(self.pods)

    def get_volume(self, volume_id, timeout):
        return {"id": volume_id, "dataCenterId": "EU-RO-1", "size": 50}

    def account(self, timeout):
        return {"clientBalance": 2.8, "currentSpendPerHr": 0.005,
                "isAutoPayEnabled": False}

    def offers(self, gpu_types, data_center, timeout):
        return [{"gpuTypeId": gpu_type, "dataCenterId": data_center,
                 "costPerHr": 0.28 + index / 10, "available": True}
                for index, gpu_type in enumerate(reversed(gpu_types))]

    def create(self, payload, timeout):
        self.actions.append(("create", payload["gpuTypeId"], payload["terminateAfter"]))
        result = self.create_results.pop(0) if self.create_results else None
        if isinstance(result, Exception):
            raise result
        pod = result or {"id": "owned", "name": payload["name"],
                         "networkVolumeId": payload["networkVolumeId"],
                         "dataCenterId": payload["dataCenterId"],
                         "gpuTypeId": payload["gpuTypeId"], "costPerHr": 0.28,
                         "createdAt": "9999-01-01T00:00:00+00:00"}
        self.pods.append(pod)
        return pod

    def terminate(self, pod_id, timeout):
        self.actions.append(("terminate", pod_id))
        self.pods = [pod for pod in self.pods if pod["id"] != pod_id]

    def wait_absent(self, pod_id, timeout):
        assert all(pod["id"] != pod_id for pod in self.pods)
        self.actions.append(("absent", pod_id))


def plan(**values):
    args = dict(episode=13, gpu_type_ids=["L4", "RTX 4000 Ada"],
                maximum_rate_usd_per_hour=0.60, total_seconds=3600,
                startup_seconds=30, shutdown_seconds=10,
                storage_quote={"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                               "source_url": STORAGE_PRICING_URL,
                               "network_volume_usd_per_gb_month": 0.07,
                               "container_storage_usd_per_gb_month": 0.10})
    args["storage_quote"]["sha256"] = digest(args["storage_quote"])
    args.update(values)
    return CapacityPlan(**args)


BASE = {"imageName": "image", "containerDiskInGb": 30}


def test_context_yields_owned_pod_budget_and_deletes_it(tmp_path):
    provider = Provider()
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["id"] == "owned"
        assert 0 < lease.work_budget_seconds <= 3590
    assert provider.actions[-2:] == [("terminate", "owned"), ("absent", "owned")]
    saved = json.loads((tmp_path / "audit" / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "RELEASED"
    assert saved["data"]["protected_pod_id"] == "781ct55zv4gkle"


def test_allowed_gpu_preference_wins_over_cheaper_reversed_offer(tmp_path):
    provider = Provider()
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["gpuTypeId"] == "L4"


def test_capacity_offer_is_polled_within_bounded_startup_window(tmp_path):
    provider = Provider()
    available = provider.offers
    calls = 0

    def delayed(gpu_types, data_center, timeout):
        nonlocal calls
        calls += 1
        return [] if calls == 1 else available(gpu_types, data_center, timeout)

    provider.offers = delayed
    with CapacityLease(
        provider,
        BASE,
        tmp_path / "audit",
        plan(),
        nonce="0123456789abcdef",
        sleep=lambda seconds: None,
    ) as lease:
        assert lease.pod["id"] == "owned"
    assert calls == 2


def test_capacity_offer_polling_stops_at_startup_deadline(tmp_path):
    provider = Provider()
    provider.offers = lambda gpu_types, data_center, timeout: []
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(IntegrityError, match="no fresh allowed"):
        with CapacityLease(
            provider,
            BASE,
            tmp_path / "audit",
            plan(total_seconds=60, startup_seconds=3, shutdown_seconds=10),
            nonce="0123456789abcdef",
            clock=lambda: now[0],
            wall_clock=lambda: 1000 + now[0],
            sleep=sleep,
        ):
            pass
    saved = json.loads((tmp_path / "audit" / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "NO_CAPACITY"
    assert now[0] == 2


def test_explicit_capacity_failure_falls_back_once(tmp_path):
    provider = Provider()
    provider.create_results = [PilotProviderError({"category": "capacity", "http_status": 500})]
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["gpuTypeId"] == "RTX 4000 Ada"
    assert [item[1] for item in provider.actions if item[0] == "create"] == ["L4", "RTX 4000 Ada"]


def test_ambiguous_create_reconciles_before_any_fallback(tmp_path):
    provider = Provider()
    original = provider.create
    def ambiguous(payload, timeout):
        pod = original(payload, timeout)
        raise TimeoutError("response lost")
    provider.create = ambiguous
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["gpuTypeId"] == "L4"
    assert len([item for item in provider.actions if item[0] == "create"]) == 1


def test_unresolved_ambiguous_create_never_falls_back(tmp_path):
    provider = Provider()
    provider.create_results = [TimeoutError("unknown")]
    ticks = iter(range(0, 1000, 5))
    lease = CapacityLease(provider, BASE, tmp_path / "audit", plan(startup_seconds=300),
                          nonce="0123456789abcdef", sleep=lambda value: None,
                          clock=lambda: next(ticks))
    with pytest.raises(IntegrityError, match="reconcile"):
        lease.acquire()
    with pytest.raises(IntegrityError, match="ownership remains unresolved"):
        lease.cleanup()
    saved = json.loads((tmp_path / "audit" / "capacity-state.json").read_text())
    assert saved["data"]["status"] in ("CREATE_REQUESTED", "AMBIGUOUS_UNRESOLVED")
    assert saved["data"]["attempts"][0]["create_error_type"] == "TimeoutError"


def test_resumed_ambiguous_create_without_visible_pod_releases_retry(tmp_path):
    provider = Provider()
    provider.create_results = [TimeoutError("unknown")]
    ticks = iter(range(0, 1000, 5))
    audit = tmp_path / "audit"
    lease = CapacityLease(
        provider,
        BASE,
        audit,
        plan(startup_seconds=300),
        nonce="0123456789abcdef",
        sleep=lambda value: None,
        clock=lambda: next(ticks),
    )
    with pytest.raises(IntegrityError, match="reconcile"):
        lease.acquire()
    actions_before_resume = list(provider.actions)

    with pytest.raises(IntegrityError, match="no externally visible Pod"):
        with CapacityLease(
            provider,
            BASE,
            audit,
            plan(),
            resume=True,
            resume_reconciliation_seconds=0,
        ):
            pass

    saved = json.loads((audit / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "NO_CAPACITY"
    assert provider.actions == actions_before_resume


def test_ambiguous_provider_error_records_only_allowlisted_safe_details(tmp_path):
    provider = Provider()
    provider.create_results = [PilotProviderError({
        "category": "http_error", "http_status": 500, "method": "POST",
        "url": "https://example.invalid/?api_key=private-key", "message": "private-key"})]
    with pytest.raises(IntegrityError, match="reconcile"):
        CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                      nonce="0123456789abcdef").acquire()
    saved_text = (tmp_path / "audit" / "capacity-state.json").read_text()
    attempt = json.loads(saved_text)["data"]["attempts"][0]
    assert attempt["create_error_type"] == "PilotProviderError"
    assert attempt["create_provider_category"] == "http_error"
    assert attempt["create_provider_http_status"] == 500
    assert attempt["create_provider_method"] == "POST"
    assert "private-key" not in saved_text
    assert "url" not in attempt and "message" not in attempt


def test_active_unrelated_pod_refuses_all_create(tmp_path):
    provider = Provider()
    provider.pods.append({"id": "other", "desiredStatus": "RUNNING"})
    with pytest.raises(IntegrityError, match="unrelated active"):
        CapacityLease(provider, BASE, tmp_path / "audit", plan()).acquire()
    assert provider.actions == []


def test_explicit_balance_reserve_is_enforced(tmp_path):
    provider = Provider()
    provider.account = lambda timeout: {"clientBalance": 1.09, "currentSpendPerHr": 0,
                                        "isAutoPayEnabled": False}
    with pytest.raises(IntegrityError, match="protected reserve"):
        CapacityLease(provider, BASE, tmp_path / "audit",
                      plan(reserve_usd=1.0, billing_margin_usd=0.10)).acquire()
    assert provider.actions == []


def test_default_plan_uses_full_balance_without_a_protected_reserve(tmp_path):
    provider = Provider()
    provider.account = lambda timeout: {"clientBalance": 0.50, "currentSpendPerHr": 0,
                                        "isAutoPayEnabled": False}
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.work_budget_seconds > 0
    saved = json.loads((tmp_path / "audit" / "capacity-state.json").read_text())
    assert saved["data"]["minimum_balance_usd"] == 0


def test_worker_failure_still_deletes_only_owned_pod(tmp_path):
    provider = Provider()
    with pytest.raises(RuntimeError, match="worker"):
        with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                           nonce="0123456789abcdef"):
            raise RuntimeError("worker")
    assert {item[1] for item in provider.actions if item[0] == "terminate"} == {"owned"}


def test_shutdown_must_verify_absent(tmp_path):
    provider = Provider()
    provider.wait_absent = lambda pod_id, timeout: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(IntegrityError, match="externally verified ABSENT"):
        with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                           nonce="0123456789abcdef"):
            pass


def test_atomic_terminate_after_is_in_every_create(tmp_path):
    provider = Provider()
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef"):
        pass
    assert provider.actions[0][2].endswith("+00:00")


def test_readiness_failure_deletes_before_next_offer(tmp_path):
    provider = Provider()
    seen = []
    def ready(pod, timeout):
        seen.append(pod["gpuTypeId"])
        if len(seen) == 1:
            raise CapacityReadinessError("SSH did not become ready")
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(), ready=ready,
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["gpuTypeId"] == "RTX 4000 Ada"
    assert provider.actions[:4] == [
        ("create", "L4", provider.actions[0][2]), ("terminate", "owned"),
        ("absent", "owned"), ("create", "RTX 4000 Ada", provider.actions[3][2])]


def test_lease_is_reduced_to_affordable_work_budget(tmp_path):
    provider = Provider()
    provider.account = lambda timeout: {"clientBalance": 1.4, "currentSpendPerHr": 0,
                                        "isAutoPayEnabled": False}
    with CapacityLease(provider, BASE, tmp_path / "audit",
                       plan(total_seconds=14400, reserve_usd=1.0, billing_margin_usd=0.10),
                       nonce="0123456789abcdef") as lease:
        assert 1700 < lease.work_budget_seconds < 1800
        assert 0 < lease.remaining_work_seconds() <= lease.work_budget_seconds


def test_provider_offer_query_is_live_and_eu_ro_1_scoped(monkeypatch):
    provider = CapacityProvider("781ct55zv4gkle", "secret")
    captured = []
    def response(request, timeout):
        captured.append(json.loads(request.data)["query"])
        return {"data": {"gpuTypes": [
            {"id": "NVIDIA L4", "lowestPrice": {"stockStatus": "High",
             "uninterruptablePrice": 0.49, "availableGpuCounts": [1]}}]}}
    monkeypatch.setattr(provider, "_json", response)
    assert provider.offers(["NVIDIA L4"], "EU-RO-1", 10) == [{
        "gpuTypeId": "NVIDIA L4", "dataCenterId": "EU-RO-1",
        "costPerHr": 0.49, "available": True}]
    assert 'dataCenterId: "EU-RO-1"' in captured[0]
    assert "secureCloud: true" in captured[0]


def test_provider_classifies_only_explicit_graphql_capacity_error(monkeypatch):
    provider = CapacityProvider("781ct55zv4gkle", "secret")
    monkeypatch.setattr(provider, "_json", lambda request, timeout: {
        "errors": [{"message": "There are not enough free GPUs to fulfill request"}]})
    with pytest.raises(PilotProviderError) as error:
        provider.create({}, 10)
    assert error.value.safe_details["category"] == "capacity"


def test_actual_rate_above_quote_is_safe_under_maximum_rate_budget(tmp_path):
    provider = Provider()
    original = provider.create
    def create(payload, timeout):
        pod = original(payload, timeout)
        pod["costPerHr"] = 0.55
        return pod
    provider.create = create
    with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                       nonce="0123456789abcdef") as lease:
        assert lease.pod["costPerHr"] == 0.55


def test_second_create_uses_refreshed_lower_balance(tmp_path):
    provider = Provider()
    balances = iter([2.8, 2.8, 1.09])
    provider.account = lambda timeout: {"clientBalance": next(balances),
                                        "currentSpendPerHr": 0.005,
                                        "isAutoPayEnabled": False}
    readiness_calls = []
    def ready(pod, timeout):
        readiness_calls.append(pod["id"])
        raise CapacityReadinessError("bounded readiness failure")
    with pytest.raises(IntegrityError, match="affordable"):
        with CapacityLease(provider, BASE, tmp_path / "audit",
                           plan(reserve_usd=1.0, billing_margin_usd=0.10), ready=ready,
                           nonce="0123456789abcdef"):
            pass
    assert len([item for item in provider.actions if item[0] == "create"]) == 1
    assert readiness_calls == ["owned"]
    assert ("absent", "owned") in provider.actions


def test_auto_pay_enabled_refuses_create(tmp_path):
    provider = Provider()
    provider.account = lambda timeout: {"clientBalance": 2.8, "currentSpendPerHr": 0,
                                        "isAutoPayEnabled": True}
    with pytest.raises(IntegrityError, match="disabled auto-pay"):
        CapacityLease(provider, BASE, tmp_path / "audit", plan()).acquire()
    assert provider.actions == []


def test_keyboard_interrupt_after_create_reconciles_and_cleans_up(tmp_path):
    provider = Provider()
    original = provider.create
    def interrupted(payload, timeout):
        original(payload, timeout)
        raise KeyboardInterrupt()
    provider.create = interrupted
    with pytest.raises(KeyboardInterrupt):
        with CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                           nonce="0123456789abcdef"):
            pass
    assert ("terminate", "owned") in provider.actions
    assert ("absent", "owned") in provider.actions


def test_resume_adopts_active_owned_pod_without_create(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    first = CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef")
    first.acquire()
    first.pod["desiredStatus"] = "RUNNING"
    provider.actions.clear()
    with CapacityLease(provider, BASE, audit, plan(), resume=True) as resumed:
        assert resumed.pod["id"] == "owned"
        assert resumed.remaining_work_seconds() > 0
    assert not [item for item in provider.actions if item[0] == "create"]
    assert provider.actions[-2:] == [("terminate", "owned"), ("absent", "owned")]


def test_resume_recorded_absent_pod_releases_without_relaunch(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    first = CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef")
    first.acquire()
    provider.pods = provider.pods[:1]
    provider.actions.clear()
    with pytest.raises(IntegrityError, match="externally ABSENT"):
        CapacityLease(provider, BASE, audit, plan(), resume=True,
                      resume_reconciliation_seconds=0).acquire()
    assert provider.actions == []
    saved = json.loads((audit / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "RELEASED"


def test_resume_rejects_wrong_state_checksum(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef").acquire()
    path = audit / "capacity-state.json"
    saved = json.loads(path.read_text())
    saved["sha256"] = "0" * 64
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(IntegrityError, match="checksum"):
        CapacityLease(provider, BASE, audit, plan(), resume=True).acquire()


def test_resume_rejects_unknown_active_pod(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    first = CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef")
    first.acquire()
    first.pod["desiredStatus"] = "RUNNING"
    provider.pods.append({"id": "intruder", "name": "other",
                          "desiredStatus": "RUNNING", "createdAt": "9999-01-01T00:00:00+00:00"})
    provider.actions.clear()
    with pytest.raises(IntegrityError, match="unrelated active"):
        CapacityLease(provider, BASE, audit, plan(), resume=True).acquire()
    assert provider.actions == []


def test_released_audit_is_preserved_before_new_lease(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    with CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef"):
        pass
    with CapacityLease(provider, BASE, audit, plan(), nonce="fedcba9876543210"):
        pass
    assert len(list(audit.glob("capacity-state.released-*.json"))) == 1


def test_cleanup_inventory_failure_cannot_report_success(tmp_path):
    provider = Provider()
    lease = CapacityLease(provider, BASE, tmp_path / "audit", plan(),
                          nonce="0123456789abcdef")
    lease.acquire()
    provider.list_pods = lambda timeout: (_ for _ in ()).throw(TimeoutError())
    lease.pod = None
    with pytest.raises(IntegrityError, match="could not be externally verified"):
        lease.cleanup()
    saved = json.loads((tmp_path / "audit" / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "UNVERIFIED"


def test_resume_same_recorded_id_with_wrong_name_is_not_absent_or_deleted(tmp_path):
    provider = Provider()
    audit = tmp_path / "audit"
    first = CapacityLease(provider, BASE, audit, plan(), nonce="0123456789abcdef")
    first.acquire()
    first.pod.update(name="different-owner", desiredStatus="RUNNING")
    provider.actions.clear()
    with pytest.raises(IntegrityError, match="could not be externally verified"):
        with CapacityLease(provider, BASE, audit, plan(), resume=True):
            pass
    assert not [item for item in provider.actions if item[0] == "terminate"]
    saved = json.loads((audit / "capacity-state.json").read_text())
    assert saved["data"]["status"] == "UNVERIFIED"


def test_load_storage_quote_requires_both_rates_and_sha(tmp_path):
    body = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": STORAGE_PRICING_URL,
            "network_volume_usd_per_gb_month": 0.07,
            "container_storage_usd_per_gb_month": 0.10}
    path = tmp_path / "quote.json"
    path.write_text(json.dumps({"data": body, "sha256": digest(body)}), encoding="utf-8")
    assert load_storage_quote(path)["sha256"] == digest(body)


def test_load_storage_quote_rejects_missing_rate(tmp_path):
    body = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": STORAGE_PRICING_URL,
            "container_storage_usd_per_gb_month": 0.10}
    path = tmp_path / "quote.json"
    path.write_text(json.dumps({"data": body, "sha256": digest(body)}), encoding="utf-8")
    with pytest.raises(IntegrityError, match="identity"):
        load_storage_quote(path)
