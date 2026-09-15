from pathlib import Path

import pytest

from mas.engine import finalize
import test_finalize as finalize_fixture


def test_finalization_accepts_real_ma_sub_checkout_name(tmp_path, monkeypatch):
    checkout = tmp_path / 'ma-sub'
    episode = checkout / 'EPISODES/Muhtemel Ask 14.Bolum'
    episode.mkdir(parents=True)
    monkeypatch.setattr(finalize, 'ROOT', checkout)
    assert finalize._validate_finalization_root(episode, 14) == episode.resolve()


def test_full_finalize_rebuilds_under_ma_sub_checkout(tmp_path, monkeypatch):
    fixture = finalize_fixture.FinalizeV2Tests()
    try:
        root, paths = fixture._workspace(str(tmp_path), project_name='ma-sub')
        fixture._inputs(root, paths)
        monkeypatch.setattr(finalize, 'mux_softsubs', fixture._fake_mux)
        report = finalize.finalize_episode_v2(**fixture._kwargs(root, paths))
        assert report['status'] == 'PASS'
        assert report['configuration_files']['series_config']['relative_path'] == 'config/production/series.yaml'
    finally:
        fixture.doCleanups()


def test_finalization_rejects_matching_episode_outside_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / 'ma-sub'
    (checkout / 'EPISODES').mkdir(parents=True)
    other = tmp_path / 'other/EPISODES/Muhtemel Ask 14.Bolum'
    other.mkdir(parents=True)
    monkeypatch.setattr(finalize, 'ROOT', checkout)
    with pytest.raises(finalize.FinalizationV2Error, match='canonical checkout'):
        finalize._validate_finalization_root(other, 14)


def test_remote_shared_episode_symlink_keeps_configuration_root_in_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / 'release'
    checkout.mkdir()
    shared = tmp_path / 'shared-episode-data'
    episode = shared / 'Muhtemel Ask 14.Bolum'
    episode.mkdir(parents=True)
    try:
        (checkout / 'EPISODES').symlink_to(shared, target_is_directory=True)
    except OSError:
        pytest.skip('Directory symlink permission unavailable')
    monkeypatch.setattr(finalize, 'ROOT', checkout)
    assert finalize._validate_finalization_root(checkout / 'EPISODES' / episode.name, 14) == episode.resolve()
    assert finalize._validate_finalization_root(episode, 14) == episode.resolve()
    config = checkout / 'config/production/series.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('subtitle: {}', encoding='utf-8')
    assert finalize._configuration_records(finalize.ROOT, {'series': config})['series']['relative_path'] == 'config/production/series.yaml'
