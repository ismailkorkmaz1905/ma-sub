import copy
import json
from pathlib import Path

import pytest
import yaml

from mas.engine.id_translation import (
    IDTranslationError, build_production_translation_policy, create_id_translation_pack,
    load_and_validate_id_translation_zip, validate_aligned_turkish_schema,
    validate_id_translation_pack, validate_production_translation_policy,
)
from mas.engine.translation_workspace import (
    REVIEW_CHECKS, collect_id_translation_workspaces, prepare_id_translation_workspaces, validate_id_workspace_output,
)
from mas.reliability import digest


def _policy():
    directory = Path(__file__).resolve().parents[2] / "config" / "production"
    values = [yaml.safe_load((directory / filename).read_text(encoding="utf-8"))
              for filename in ("series.yaml", "names.yaml", "religious_terms.yaml")]
    return build_production_translation_policy(*values)


def _schema(texts=None, policy=None):
    texts = texts or ["Merhaba.", "Nasılsın?", "Ben iyiyim.", "Buraya gel.", "Bekle.", "Gidelim."]
    return validate_aligned_turkish_schema({
        "schema_version": "2.0", "episode": 14, "production_policy": policy or _policy(),
        "blocks": [{"block_uid": f"cue-{index}", "block_index": index + 1,
                    "start_ms": index * 30_000, "end_ms": index * 30_000 + 2500, "tr_text": text,
                    "alignment_provenance": {"timing_source": "whisperx_ctc_forced_alignment"}}
                   for index, text in enumerate(texts)],
    })


def _pack(tmp_path, schema=None, filename="input.zip"):
    schema = schema or _schema()
    pack = tmp_path / filename
    create_id_translation_pack(schema, pack, glossary=schema["production_policy"]["glossary"])
    return pack


def _inputs(workspace):
    manifest = json.loads(workspace.read_text(encoding="utf-8"))["payload"]
    return [json.loads((workspace.parent / worker["input_file"]).read_text(encoding="utf-8"))
            for worker in manifest["workers"]]


def _returns(workspace):
    translations = ["Halo.", "Apa kabar?", "Aku baik.", "Kemari.", "Tunggu.", "Ayo pergi."]
    return {value["worker_id"]: [{**record, "id_final": translations[record["block_index"] - 1]}
                                  for record in value["records"]]
            for value in _inputs(workspace)}


def _reviews(result):
    return [{"block_uid": item["block_uid"], "source_record_sha256": item["source_record_sha256"],
             "translation_sha256": item["translation_sha256"], "context_sha256": item["context_sha256"],
             "policy_sha256": item["policy_sha256"], "reviewer": "independent-local-test-reviewer",
             "checks": {name: True for name in REVIEW_CHECKS}}
            for item in result["risk_review_items"]]


def _collect_reviewed(pack, workspace, returned, **kwargs):
    pending = collect_id_translation_workspaces(pack, workspace, returned)
    reviews = [review for review in _reviews(pending) if review["block_uid"] in pending["review_pending_uids"]]
    return collect_id_translation_workspaces(pack, workspace, returned, reviews=reviews, **kwargs)


def test_policy_hash_binds_configuration_without_relaxing_limits():
    policy = _policy()
    assert validate_production_translation_policy(policy) == policy
    changed = copy.deepcopy(policy)
    changed["timing_qa"]["maximum_cps"] = 21
    with pytest.raises(IDTranslationError, match="binding changed"):
        validate_production_translation_policy(changed)
    changed = copy.deepcopy(policy["series"])
    changed["subtitle"]["qa_max_chars_per_line"] = 84
    with pytest.raises(IDTranslationError, match="weaken"):
        build_production_translation_policy(changed, policy["names"], policy["religious"])


def test_changed_policy_changes_schema_and_rejects_old_pack(tmp_path):
    schema = _schema()
    pack = _pack(tmp_path, schema)
    policy = schema["production_policy"]
    series = copy.deepcopy(policy["series"])
    series["subtitle"]["preferred_max_cps"] = 19
    changed = _schema(policy=build_production_translation_policy(series, policy["names"], policy["religious"]))
    assert changed["schema_sha256"] != schema["schema_sha256"]
    with pytest.raises(IDTranslationError, match="differs from expected schema"):
        validate_id_translation_pack(pack, expected_schema=changed)


def test_pack_cannot_override_frozen_glossary(tmp_path):
    schema = _schema()
    glossary = copy.deepcopy(schema["production_policy"]["glossary"])
    glossary["canonical_names"].append("Added Name")
    with pytest.raises(IDTranslationError, match="frozen production policy"):
        create_id_translation_pack(schema, tmp_path / "bad.zip", glossary=glossary)


def test_three_deterministic_uid_scopes_and_readonly_context(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    original = workspace.read_bytes()
    assert prepare_id_translation_workspaces(pack, tmp_path / "work") == workspace
    assert workspace.read_bytes() == original
    inputs = _inputs(workspace)
    assert len(inputs) == 3
    assert [uid for value in inputs for uid in value["owned_uids"]] == [f"cue-{index}" for index in range(6)]
    for value in inputs:
        assert not set(value["owned_uids"]) & {record["block_uid"] for record in value["read_only_context"]}
        assert "Never move words or" in value["instructions"]
        assert "never return it" in value["instructions"]


def test_complete_workspace_emits_original_exact_zip_contract(tmp_path):
    schema = _schema()
    pack = _pack(tmp_path, schema)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    result = _collect_reviewed(pack, workspace, _returns(workspace), out_zip=tmp_path / "return.zip")
    assert result["status"] == "EXACT_RETURN_CREATED"
    assert validate_id_workspace_output(pack, tmp_path / "return.zip") == result
    validation = load_and_validate_id_translation_zip(schema, tmp_path / "return.zip",
                                                      input_manifest=validate_id_translation_pack(pack))
    assert validation.ok and validation.output_block_count == 6


def test_only_failed_record_is_reissued_while_good_translations_are_retained(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    returned = _returns(workspace)
    returned["worker-02"][0]["id_final"] = ""
    result = _collect_reviewed(pack, workspace, returned)
    assert result["rejected_uids"] == ["cue-2"]
    retry = prepare_id_translation_workspaces(pack, tmp_path / "work", previous_result=result)
    assert [uid for value in _inputs(retry) for uid in value["owned_uids"]] == ["cue-2"]
    complete = _collect_reviewed(pack, retry, _returns(retry))
    assert complete["status"] == "READY_SCOPE"
    assert len(complete["accepted_records"]) == 6


def test_source_change_reissues_only_changed_cue_and_revalidates_reused_drafts(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    result = _collect_reviewed(pack, workspace, _returns(workspace))
    texts = ["Merhaba.", "Nasılsın?", "Ben çok iyiyim.", "Buraya gel.", "Bekle.", "Gidelim."]
    changed_pack = _pack(tmp_path, _schema(texts), "changed.zip")
    changed = prepare_id_translation_workspaces(changed_pack, tmp_path / "work", previous_result=result)
    assert [uid for value in _inputs(changed) for uid in value["owned_uids"]] == ["cue-2"]
    assert json.loads(changed.read_text(encoding="utf-8"))["payload"]["freeze"] != result["freeze"]
    assert [review["block_uid"] for review in json.loads(changed.read_text(encoding="utf-8"))["payload"]["seed_reviews"]] == ["cue-5"]


def test_first_scope_cannot_emit_incomplete_full_episode_zip(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work", scope_uids=["cue-0", "cue-1"])
    result = _collect_reviewed(pack, workspace, _returns(workspace))
    assert result["status"] == "READY_SCOPE" and len(result["accepted_records"]) == 2
    with pytest.raises(IDTranslationError, match="complete scope"):
        collect_id_translation_workspaces(pack, workspace, _returns(workspace), out_zip=tmp_path / "bad.zip")


def test_worker_cannot_submit_neighbor_context_or_reorder_owned_cues(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    returned = _returns(workspace)
    returned["worker-01"].append(returned["worker-02"][0])
    with pytest.raises(IDTranslationError, match="unowned/context"):
        collect_id_translation_workspaces(pack, workspace, returned)
    returned = _returns(workspace)
    returned["worker-01"].reverse()
    with pytest.raises(IDTranslationError, match="reordered"):
        collect_id_translation_workspaces(pack, workspace, returned)


def test_changed_context_and_nested_immutable_boolean_are_rejected(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    returned = _returns(workspace)
    returned["worker-01"][0]["alignment_provenance"]["timing_source"] = "invented"
    result = collect_id_translation_workspaces(pack, workspace, returned)
    assert result["rejections"]["cue-0"] == ["immutable_field_changed"]
    path = workspace.parent / "worker-01" / "input.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["read_only_context"][0]["tr_text"] = "Unrelated scene."
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(IDTranslationError, match="read-only context changed"):
        collect_id_translation_workspaces(pack, workspace, _returns(workspace))


def test_religious_failure_cannot_be_approved_and_corrected_risk_needs_bound_review(tmp_path):
    pack = _pack(tmp_path, _schema(["Allah aşkına, Defne."]))
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    returned = _returns(workspace)
    record = next(record for batch in returned.values() for record in batch)
    record["id_final"] = "Semoga, Defne."
    failed = collect_id_translation_workspaces(pack, workspace, returned)
    assert failed["rejected_uids"] == ["cue-0"]
    record["id_final"] = "Demi Allah, Defne."
    pending = collect_id_translation_workspaces(pack, workspace, returned)
    assert pending["review_pending_uids"] == ["cue-0"]
    assert set(pending["risk_review_items"][0]["risk_tags"]) == {"proper_names", "religious_meaning"}
    review = _reviews(pending)
    forged = copy.deepcopy(review)
    forged[0]["translation_sha256"] = digest("different text")
    with pytest.raises(IDTranslationError, match="review binding"):
        collect_id_translation_workspaces(pack, workspace, returned, reviews=forged)
    complete = collect_id_translation_workspaces(pack, workspace, returned, reviews=review, out_zip=tmp_path / "return.zip")
    assert complete["status"] == "EXACT_RETURN_CREATED"
    retry = prepare_id_translation_workspaces(pack, tmp_path / "work", previous_result=complete)
    assert all(not value["owned_uids"] for value in _inputs(retry))
    assert collect_id_translation_workspaces(pack, retry, {})["status"] == "READY_SCOPE"


def test_workspace_output_receipt_detects_zip_and_review_tampering(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    output = tmp_path / "return.zip"
    _collect_reviewed(pack, workspace, _returns(workspace), out_zip=output)
    original = output.read_bytes()
    output.write_bytes(original + b"changed")
    with pytest.raises(IDTranslationError, match="receipt binding changed"):
        validate_id_workspace_output(pack, output)
    output.write_bytes(original)
    receipt = Path(str(output) + ".workspace.json")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["reviews"] = []
    value.pop("result_sha256")
    value["result_sha256"] = digest(value)
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(IDTranslationError, match="meaning/register review"):
        validate_id_workspace_output(pack, output)


def test_failed_meaning_review_reissues_only_that_cue(tmp_path):
    pack = _pack(tmp_path)
    workspace = prepare_id_translation_workspaces(pack, tmp_path / "work")
    returned = _returns(workspace)
    pending = collect_id_translation_workspaces(pack, workspace, returned)
    reviews = _reviews(pending)
    reviews[2]["checks"]["cue_ownership"] = False
    result = collect_id_translation_workspaces(pack, workspace, returned, reviews=reviews)
    assert result["rejected_uids"] == ["cue-2"]
    retry = prepare_id_translation_workspaces(pack, tmp_path / "work", previous_result=result)
    assert [uid for value in _inputs(retry) for uid in value["owned_uids"]] == ["cue-2"]
    assert len(json.loads(retry.read_text(encoding="utf-8"))["payload"]["seed_reviews"]) == 5
