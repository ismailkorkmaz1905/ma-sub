import copy

import pytest

from mas.engine.id_translation import (build_id_translation_records,
    create_id_translation_output_zip, create_id_translation_pack,
    load_default_id_translation_glossary)
from mas.id_return_preflight import preflight_local_id_return


def schema(episode=11):
    return {"schema_version": "2.0", "episode": episode, "block_count": 1,
            "pipeline": "correct-tr-then-align-then-id", "blocks": [{
                "block_uid": f"MA{episode}-V2-000001-abcdef01", "block_index": 1,
                "start_ms": 1000, "end_ms": 3000, "tr_text": "Merhaba.",
                "alignment_provenance": {"timing_source": "whisperx_forced_alignment",
                    "alignment_model": "tr-test-model", "whisperx_version": "test",
                    "first_word_start_ms": 1050, "last_word_end_ms": 2900}}]}


def make_files(root, *, episode=11, text="Halo.", review=False, pack_episode=None):
    name = f"Muhtemel Ask {episode}.Bolum"
    pack = root / "translation_input" / f"{name}_ID_TRANSLATION_PACK.zip"
    returned = root / "translation_output" / f"{name}_ID_TRANSLATED.zip"
    source = schema(pack_episode or episode)
    manifest = create_id_translation_pack(source, pack, batch_size=1,
                                          glossary=load_default_id_translation_glossary())
    records = build_id_translation_records(source)
    records[0].update(id_final=text, review_required=review,
                      note="check" if review else "")
    create_id_translation_output_zip(source, records, returned, input_manifest=manifest)
    return pack, returned


def test_absent_return_is_noop(tmp_path):
    assert preflight_local_id_return(tmp_path, 11, "config/production") is None


def test_valid_return_passes_and_persists_hashes(tmp_path):
    pack, returned = make_files(tmp_path)
    report = preflight_local_id_return(tmp_path, 11, "config/production")
    assert report["status"] == "PASS"
    assert report["pack_sha256"] and report["return_sha256"]


@pytest.mark.parametrize("kind", ["long", "review"])
def test_readability_and_review_required_fail_before_compute(tmp_path, kind):
    make_files(tmp_path, text="kata " * 100 if kind == "long" else "Halo.",
               review=kind == "review")
    with pytest.raises(Exception):
        preflight_local_id_return(tmp_path, 11, "config/production")
    report = (tmp_path / "work/id_return_local_preflight.json").read_text()
    assert '"status": "FAIL"' in report
    if kind == "review":
        assert '"review_required_uids"' in report
        assert '"MA11-V2-000001-abcdef01"' in report


def test_hard_id_cps_fails_when_layout_still_fits(tmp_path):
    make_files(tmp_path, text="a" * 50)
    with pytest.raises(Exception, match="failed subtitle QA"):
        preflight_local_id_return(tmp_path, 11, "config/production")
    report = (tmp_path / "work/id_return_local_preflight.json").read_text()
    assert '"high_cps_id_count": 1' in report
    assert '"block_uid": "MA11-V2-000001-abcdef01"' in report


def test_return_without_pack_fails(tmp_path):
    pack, returned = make_files(tmp_path)
    pack.unlink()
    with pytest.raises(Exception, match="without its input pack"):
        preflight_local_id_return(tmp_path, 11, "config/production")


def test_wrong_episode_pack_fails(tmp_path):
    make_files(tmp_path, pack_episode=12)
    with pytest.raises(Exception, match="episode mismatch"):
        preflight_local_id_return(tmp_path, 11, "config/production")


def test_pack_mutation_during_checks_fails(tmp_path, monkeypatch):
    import mas.id_return_preflight as module
    pack, returned = make_files(tmp_path)
    original = module.load_and_validate_id_translation_zip
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        pack.write_bytes(pack.read_bytes() + b"changed")
        return result
    monkeypatch.setattr(module, "load_and_validate_id_translation_zip", mutate)
    with pytest.raises(Exception, match="changed during local preflight"):
        preflight_local_id_return(tmp_path, 11, "config/production")
