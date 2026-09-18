import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas import cli, runpod_controller as rc
from mas.reliability import BudgetExceeded, IntegrityError, atomic_json, digest
from mas.runpod_capacity import CapacityLease, CapacityPlan
from test_runpod_capacity import BASE, Provider, plan


@pytest.fixture
def active(tmp_path, monkeypatch):
    root = tmp_path / 'episode'
    audit = root / 'work/capacity/original'
    provider = Provider()
    first = CapacityLease(provider, BASE, audit, plan(episode=14), nonce='0123456789abcdef')
    first.acquire()
    image = 'example.invalid/runtime@sha256:' + 'a' * 64
    first.pod.update(desiredStatus='RUNNING', imageName=image, publicIp='fixture', portMappings={'22': '22'})
    atomic_json(root / 'work/controller-lease.json', {'data': {
        'commit': 'a' * 40, 'source_url': 'https://example.invalid/14',
        'audit': 'work/capacity/original'}, 'sha256': digest({
        'commit': 'a' * 40, 'source_url': 'https://example.invalid/14',
        'audit': 'work/capacity/original'})})
    key = tmp_path / 'ssh-key'
    key.write_text('fixture key')
    key.with_suffix('.pub').write_text('ssh-ed25519 fixture')
    quote = tmp_path / 'quote.json'
    quote_body = {k: v for k, v in plan().storage_quote.items() if k != 'sha256'}
    atomic_json(quote, {'data': quote_body, 'sha256': digest(quote_body)})
    monkeypatch.setenv('RUNPOD_API_KEY', 'test-key-not-live')
    monkeypatch.setenv('MAS_RUNPOD_IMAGE', image)
    monkeypatch.setenv('MAS_RUNPOD_STORAGE_QUOTE', str(quote))
    monkeypatch.setenv('MAS_RUNPOD_GPU_TYPE_IDS', 'L4|RTX 4000 Ada')
    monkeypatch.setenv('MAS_EPISODE_BUDGET_SECONDS', '21600')
    monkeypatch.setattr(rc, 'ROOT', tmp_path)
    monkeypatch.setattr(rc, 'episode_dir', lambda episode: root)
    monkeypatch.setattr(rc, 'CapacityProvider', lambda *args: provider)
    provider.get_pod = lambda pod_id, timeout: next(p for p in provider.pods if p['id'] == pod_id)
    monkeypatch.setattr(rc, '_required_environment', lambda: {
        'RUNPOD_POD_ID': '781ct55zv4gkle', 'RUNPOD_API_KEY': 'test-key-not-live',
        'MAS_RUNPOD_SSH_KEY': str(key), 'MAS_DRIVE_STRICT_REMOTE': 'gdrive:fixture'})
    monkeypatch.setattr(rc, '_local_preflight', lambda values: ('a' * 40, tmp_path / 'rclone.conf'))
    monkeypatch.setattr(rc, '_prepare_official_source', lambda *args: 'https://example.invalid/14')
    monkeypatch.setattr(rc, 'RunPodClient', lambda *args, **kw: SimpleNamespace(wait=lambda *a, **k: first.pod))
    monkeypatch.setattr(rc, '_wait_for_ssh', lambda *args, **kwargs: None)
    monkeypatch.setattr(rc, '_run_remote_session', lambda *args, **kwargs: 0)
    monkeypatch.setattr(rc, '_drain_notifications', lambda *args: None)
    provider.actions.clear()
    return root, audit, provider, first, quote, key


def set_remaining(root, seconds):
    start = datetime.now(timezone.utc) - timedelta(seconds=21600 - seconds)
    body = {'episode': 14, 'started_at': start.isoformat(), 'limit_seconds': 21600,
            'excluded_wait_seconds': 0}
    atomic_json(root / 'work/controller_budget.json', {'data': body, 'sha256': digest(body)})
    return start.isoformat()


def assert_released(audit, provider):
    state = json.loads((audit / 'capacity-state.json').read_text())['data']
    assert state['status'] == 'RELEASED'
    assert state['owned_pod_ids'] == ['owned']
    assert state['shutdown'][0]['status'] == 'ABSENT'
    assert not any(a[0] == 'create' for a in provider.actions)
    assert ('terminate', '781ct55zv4gkle') not in provider.actions
    assert ('absent', 'owned') in provider.actions


@pytest.mark.parametrize('seconds', [421, 420.9, 420, 300, 122])
def test_existing_lease_can_resume_below_new_start_threshold(active, seconds):
    root, audit, provider, *_ = active
    started = set_remaining(root, seconds)
    assert rc._run_remote_episode_once(14) == 0
    assert_released(audit, provider)
    assert json.loads((root / 'work/controller_budget.json').read_text())['data']['started_at'] == started


@pytest.mark.parametrize('seconds', [120, 1, -1])
def test_shutdown_reserve_or_expired_budget_releases_without_create(active, seconds):
    root, audit, provider, *_ = active
    started = set_remaining(root, seconds)
    with pytest.raises((ValueError, BudgetExceeded)):
        rc._run_remote_episode_once(14)
    assert_released(audit, provider)
    assert json.loads((root / 'work/controller_budget.json').read_text())['data']['started_at'] == started


@pytest.mark.parametrize('failure', ['stale_quote', 'missing_quote', 'image', 'public_key', 'dirty', 'environment'])
@pytest.mark.parametrize('remaining', [300, -1])
def test_controller_preflight_errors_cannot_strand_existing_capacity(active, monkeypatch, failure, remaining):
    root, audit, provider, _, quote, key = active
    started = set_remaining(root, remaining)
    if failure == 'stale_quote':
        saved = json.loads(quote.read_text())
        saved['data']['observed_at_utc'] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        saved['sha256'] = digest(saved['data'])
        atomic_json(quote, saved)
    elif failure == 'missing_quote':
        monkeypatch.delenv('MAS_RUNPOD_STORAGE_QUOTE')
    elif failure == 'image':
        monkeypatch.setenv('MAS_RUNPOD_IMAGE', 'mutable:latest')
    elif failure == 'public_key':
        key.with_suffix('.pub').unlink()
    elif failure == 'dirty':
        monkeypatch.setattr(rc, '_local_preflight', lambda values: (_ for _ in ()).throw(
            rc.RunPodControllerError('repository must be clean')))
    else:
        monkeypatch.setattr(rc, '_required_environment', lambda: (_ for _ in ()).throw(
            rc.RunPodControllerError('missing local environment')))
    with pytest.raises((rc.RunPodControllerError, IntegrityError, FileNotFoundError, BudgetExceeded)):
        rc._run_remote_episode_once(14)
    assert_released(audit, provider)
    assert json.loads((root / 'work/controller_budget.json').read_text())['data']['started_at'] == started


def test_cleanup_only_ignores_other_active_pods_and_preserves_budget(active):
    root, audit, provider, *_ = active
    set_remaining(root, -1)
    before = (root / 'work/controller_budget.json').read_bytes()
    provider.pods.append({'id': 'unrelated', 'name': 'someone-else', 'desiredStatus': 'RUNNING'})
    with pytest.raises(RuntimeError, match='interrupted'):
        with rc._cleanup_on_controller_failure(root, 14):
            raise RuntimeError('interrupted')
    assert_released(audit, provider)
    assert any(p['id'] == 'unrelated' for p in provider.pods)
    assert (root / 'work/controller_budget.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['checksum', 'foreign_name', 'missing_api_key', 'inventory_timeout'])
def test_unverified_cleanup_never_claims_pass_or_deletes_unknown_pod(active, monkeypatch, kind):
    root, audit, provider, first, *_ = active
    if kind == 'checksum':
        saved = json.loads((audit / 'capacity-state.json').read_text())
        saved['sha256'] = '0' * 64
        atomic_json(audit / 'capacity-state.json', saved)
    elif kind == 'foreign_name':
        first.pod['name'] = 'somebody-elses-pod'
    elif kind == 'missing_api_key':
        monkeypatch.delenv('RUNPOD_API_KEY')
    else:
        provider.list_pods = lambda timeout: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(rc.RunPodCleanupRequired, match='EXTERNAL CLEANUP REQUIRED'):
        with rc._cleanup_on_controller_failure(root, 14):
            raise RuntimeError('local preflight failed')
    assert not provider.actions
    assert any(p['id'] == 'owned' for p in provider.pods)


def test_cleanup_still_runs_after_capacity_work_deadline(active):
    _, audit, provider, lease, *_ = active
    lease._deadline = lease.clock() - 1
    lease.cleanup()
    assert_released(audit, provider)


def test_lost_terminate_response_requires_and_accepts_external_absence(active):
    _, audit, provider, lease, *_ = active
    terminate = provider.terminate
    def lost(pod_id, timeout):
        terminate(pod_id, timeout)
        raise TimeoutError('lost response')
    provider.terminate = lost
    lease.cleanup()
    assert_released(audit, provider)


def test_resume_plan_cannot_be_used_to_acquire_new_capacity(tmp_path):
    provider = Provider()
    with pytest.raises(IntegrityError, match='cannot acquire'):
        CapacityLease(provider, BASE, tmp_path, plan(total_seconds=20, startup_seconds=30,
                                                    shutdown_seconds=10, resume=True)).acquire()
    assert not provider.actions


@pytest.mark.parametrize('missing', ['ffmpeg', 'rclone', 'torch'])
def test_inventory_doctor_returns_failure_for_missing_dependency(monkeypatch, capsys, missing):
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None if name == missing else name)
    monkeypatch.setitem(sys.modules, 'torch', None if missing == 'torch' else SimpleNamespace(
        __version__='fixture', cuda=SimpleNamespace(is_available=lambda: False)))
    assert cli.doctor() == 1
    assert 'not production readiness' in capsys.readouterr().out


def test_windows_launcher_refreshes_new_required_settings():
    source = (Path(rc.__file__).resolve().parents[2] / 'mas.ps1').read_text()
    names = source.split('$userEnvironmentNames = @(', 1)[1].split(')', 1)[0]
    assert '"MAS_RUNPOD_IMAGE"' in names
    assert '"MAS_RUNPOD_REGISTRY_AUTH_ID"' in names
    assert '"MAS_RUNPOD_STORAGE_QUOTE"' in names
    assert 'elseif (-not [Environment]::GetEnvironmentVariable($name, "Process"))' in source


def test_large_immutable_image_gets_full_bounded_startup_window():
    source = Path(rc.__file__).read_text(encoding='utf-8')
    assert 'startup_seconds=900' in source
    assert 'remaining = min(900, remaining)' in source
    assert 'remaining = min(90, remaining)' not in source


def test_controller_doctor_is_no_compute_and_does_not_require_local_cuda(active, monkeypatch, capsys):
    root, _, provider, *_ = active
    monkeypatch.setattr(cli.shutil, 'which', lambda name: name)
    monkeypatch.setattr(rc, 'drive_preflight', lambda *args, **kwargs: {'free_bytes': 100_000_000_000})
    from mas.engine import burned_mp4
    monkeypatch.setattr(burned_mp4, '_qsv_hardware', lambda **kwargs: {'qsv_hardware_probe': 'PASS'})
    monkeypatch.setitem(sys.modules, 'torch', None)
    before = sorted(str(p) for p in root.rglob('*'))
    assert cli.doctor(controller=True) == 0
    assert not provider.actions
    assert before == sorted(str(p) for p in root.rglob('*'))
    assert 'no Pod acquired' in capsys.readouterr().out


def test_cleanup_required_cli_never_recommends_relaunch(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext
    monkeypatch.setattr(cli, 'RunLog', lambda *args: nullcontext())
    monkeypatch.setattr(cli, 'episode_dir', lambda _: tmp_path)
    monkeypatch.setattr(cli, 'run_remote_episode', lambda *a: (_ for _ in ()).throw(
        rc.RunPodCleanupRequired('EXTERNAL CLEANUP REQUIRED')))
    notices = []
    monkeypatch.setattr(cli, 'enqueue_notification', lambda *a, **k: notices.append(a[2]))
    monkeypatch.setattr(cli, 'drain_cli_notifications', lambda *a: None)
    assert cli.main(['run', '14']) == 1
    assert 'DO NOT RELAUNCH' in capsys.readouterr().err
    assert len(notices) == 1 and './mas run 14' not in notices[0]
