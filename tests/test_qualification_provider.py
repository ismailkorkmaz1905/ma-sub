import json

import pytest

from mas.reliability import IntegrityError
from mas.subtitle.pilot_runner import PilotProviderError
from mas.subtitle.qualification_provider import QualificationProvider


def test_create_sends_atomic_termination_schedule_and_reads_full_identity(monkeypatch):
    provider = QualificationProvider("protected", "secret")
    calls = []
    payload = {"terminateAfter": "2026-09-06T07:00:00+00:00"}

    def response(request, timeout):
        calls.append(request)
        return {"data": {"podFindAndDeployOnDemand": {"id": "new-pod"}}}

    monkeypatch.setattr(provider, "_json", response)
    monkeypatch.setattr(provider, "get_pod", lambda pod_id, timeout: {"id": pod_id})
    assert provider.create(payload, 20) == {"id": "new-pod"}
    assert len(calls) == 1
    assert json.loads(calls[0].data)["variables"]["input"] == payload


def test_ambiguous_create_is_never_retried(monkeypatch):
    provider = QualificationProvider("protected", "secret")
    calls = []

    def timeout(*args):
        calls.append(1)
        raise TimeoutError()

    monkeypatch.setattr(provider, "_json", timeout)
    with pytest.raises(TimeoutError):
        provider.create({}, 20)
    assert calls == [1]


def test_absence_requires_404_not_provider_failure(monkeypatch):
    provider = QualificationProvider("protected", "secret")
    def fail(pod_id, timeout):
        raise PilotProviderError({"http_status": 500})
    monkeypatch.setattr(provider, "get_pod", fail)
    with pytest.raises(PilotProviderError):
        provider.wait_absent("new", 1)
    def absent(pod_id, timeout):
        raise PilotProviderError({"http_status": 404})
    monkeypatch.setattr(provider, "get_pod", absent)
    provider.wait_absent("new", 1)


def test_protected_pod_cannot_be_terminated():
    with pytest.raises(IntegrityError, match="protected"):
        QualificationProvider("protected", "secret").terminate("protected", 1)
