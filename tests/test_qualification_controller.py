import pytest

from mas.reliability import IntegrityError
from mas.subtitle.qualification_controller import run_disposable_qualification


BASE = {"networkVolumeId": "xgogcmey5o", "imageName": "image",
        "dataCenterId": "EU-RO-1", "gpuTypeId": "GPU"}


def pod(id="new", rate=0.5):
    return {"id": id, "name": "mas-ep11-qualification-0123456789abcdef",
            "networkVolumeId": "xgogcmey5o", "imageName": "image",
            "dataCenterId": "EU-RO-1", "gpuTypeId": "GPU",
            "createdAt": "9999-01-01 00:00:00.000 +0000 UTC", "costPerHr": rate}


class Provider:
    def __init__(self, created=None, listed=None):
        self.created, self.listed, self.actions, self.list_calls = created, listed or [], [], 0
    def list_pods(self, timeout):
        self.list_calls += 1
        return [{"id": "old", "desiredStatus": "EXITED"}, *(self.listed if self.list_calls > 1 else [])]
    def account(self, timeout):
        return {"clientBalance": 3.7, "currentSpendPerHr": 0.005,
                "isAutoPayEnabled": False}
    def get_volume(self, id, timeout):
        return {"id": id, "dataCenterId": "EU-RO-1"}
    def create(self, payload, timeout): self.actions.append("create"); return self.created
    def terminate(self, id, timeout): self.actions.append(("terminate", id))
    def wait_absent(self, id, timeout): self.actions.append(("absent", id))


def run(tmp_path, provider, worker=lambda pod, timeout: "done", **kwargs):
    return run_disposable_qualification(provider, BASE, worker, tmp_path / "out",
                                        maximum_rate_usd_per_hour=0.6,
                                        protected_pod_id="old",
                                        nonce="0123456789abcdef", sleep=lambda value: None,
                                        **kwargs)


def test_success_terminates_only_new_exact_pod(tmp_path):
    provider = Provider(created=pod())
    assert run(tmp_path, provider) == "done"
    assert provider.actions == ["create", ("terminate", "new"), ("absent", "new")]


def test_active_protected_pod_prevents_creation(tmp_path):
    provider = Provider(created=pod())
    provider.list_pods = lambda timeout: [{"id": "old", "desiredStatus": "RUNNING"}]
    with pytest.raises(IntegrityError, match="must remain EXITED"):
        run(tmp_path, provider)
    assert provider.actions == []


def test_ambiguous_create_reconciles_without_second_post(tmp_path):
    provider = Provider(created=None, listed=[pod()])
    assert run(tmp_path, provider) == "done"
    assert provider.actions.count("create") == 1


def test_old_id_collision_is_never_terminated(tmp_path):
    provider = Provider(created=pod("old"))
    with pytest.raises(IntegrityError, match="resolve one"):
        run(tmp_path, provider, reconciliation_seconds=0)
    assert provider.actions == ["create"]


def test_price_over_cap_still_terminates(tmp_path):
    provider = Provider(created=pod(rate=0.61))
    with pytest.raises(IntegrityError, match="rate exceeds"):
        run(tmp_path, provider)
    assert ("absent", "new") in provider.actions


def test_parent_worker_exception_still_terminates(tmp_path):
    provider = Provider(created=pod())
    with pytest.raises(RuntimeError, match="worker"):
        run(tmp_path, provider, worker=lambda pod, timeout: (_ for _ in ()).throw(RuntimeError("worker")))
    assert ("absent", "new") in provider.actions


def test_failed_shutdown_overrides_success(tmp_path):
    provider = Provider(created=pod())
    provider.wait_absent = lambda id, timeout: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(IntegrityError, match="termination was not verified"):
        run(tmp_path, provider)


def test_actual_rest_created_at_format_is_owned(tmp_path):
    provider = Provider(created=pod())
    assert run(tmp_path, provider) == "done"
    assert ("terminate", "new") in provider.actions


def test_zero_existing_account_rate_is_valid(tmp_path):
    provider = Provider(created=pod())
    provider.account = lambda timeout: {"clientBalance": 3.7,
                                        "currentSpendPerHr": 0,
                                        "isAutoPayEnabled": False}
    assert run(tmp_path, provider) == "done"


def test_pilot_limit_is_not_extended_by_episode_support(tmp_path):
    provider = Provider(created=pod())
    with pytest.raises(ValueError, match="1..550"):
        run(tmp_path, provider, lease_seconds=551)
    assert provider.actions == []


@pytest.mark.parametrize('episode', [12, 13, 27])
def test_episode_lease_preserves_budget_and_external_cleanup(tmp_path, episode):
    from datetime import datetime, timedelta, timezone
    from mas.subtitle.qualification_controller import run_disposable_episode
    created = pod()
    created['name'] = f'mas-ep{episode}-draft-0123456789abcdef'
    provider = Provider(created=created)
    deadline = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    assert run_disposable_episode(provider, BASE, lambda pod, timeout: 'done', tmp_path / 'out',
        episode=episode, deadline_utc=deadline, lease_seconds=5400,
        maximum_rate_usd_per_hour=.6, protected_pod_id='old',
        nonce='0123456789abcdef') == 'done'
    assert provider.actions == ['create', ('terminate', 'new'), ('absent', 'new')]


def test_episode_cannot_spend_balance_reserve(tmp_path):
    from datetime import datetime, timedelta, timezone
    from mas.subtitle.qualification_controller import run_disposable_episode
    provider = Provider(created=pod())
    provider.account = lambda timeout: {'clientBalance':1.2, 'currentSpendPerHr':.005,
                                       'isAutoPayEnabled':False}
    deadline = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    with pytest.raises(IntegrityError, match='breach balance reserve'):
        run_disposable_episode(provider, BASE, lambda pod, timeout: None, tmp_path / 'out',
            episode=12, deadline_utc=deadline, lease_seconds=5400,
            maximum_rate_usd_per_hour=.6, protected_pod_id='old')
    assert provider.actions == []
