import copy
import hashlib
import re
from pathlib import Path

from ..reliability import atomic_json, digest, read_json
from .id_translation import (
    ID_OUTPUT_FIELDS, ID_TRANSLATION_INSTRUCTIONS, IDTranslationError,
    _json_clone, _json_equal, _open_checked_zip, _parse_json, build_id_translation_records,
    create_id_translation_output_zip, validate_aligned_turkish_schema,
    load_and_validate_id_translation_zip, validate_id_translation_pack, validate_id_translation_records,
)
from .subtitle_qa import run_subtitle_qa


WORKSPACE_INSTRUCTIONS = """This is an offline, UID-owned translation assignment.
Translate only records in owned_uids, in their original order. read_only_context
is context only: never return it, edit it, borrow its words, or move meaning to
another cue. Return immutable record fields plus only id_final, review_required
and note. Do not run a translation API or machine-translate the entire episode.
Uncertain meaning must remain review_required; never invent a missing line.
Rule checks do not establish natural Indonesian or semantic equivalence. The
coordinator issues separate, hash-bound meaning/register review for risky cues.
"""
REVIEW_CHECKS = ("meaning", "natural_indonesian", "register", "cue_ownership")


def _pack(path):
    path = Path(path)
    manifest = validate_id_translation_pack(path)
    with _open_checked_zip(path) as archive:
        schema = validate_aligned_turkish_schema(_parse_json(archive.read("schema.json"), member="schema.json"))
    policy = schema.get("production_policy")
    if policy is None:
        raise IDTranslationError("Parallel workspaces require a frozen production policy")
    freeze = {"input_pack_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "manifest_sha256": digest(manifest), "schema_sha256": schema["schema_sha256"],
              "policy_sha256": policy["policy_sha256"],
              "workspace_instructions_sha256": digest(WORKSPACE_INSTRUCTIONS)}
    return schema, manifest, build_id_translation_records(schema), freeze


def _checked_record(expected, record, policy):
    if not isinstance(record, dict):
        return None, ["record_not_object"]
    if set(record) - set(expected) - ID_OUTPUT_FIELDS or set(expected) - set(record):
        return None, ["immutable_field_set"]
    if any(not _json_equal(record[name], value) for name, value in expected.items()):
        return None, ["immutable_field_changed"]
    text = record.get("id_final")
    if not isinstance(text, str) or not text.strip():
        return None, ["blank_translation"]
    if type(record.get("review_required", False)) is not bool or not isinstance(record.get("note", ""), str):
        return None, ["invalid_review_fields"]
    normalized = {**copy.deepcopy(expected), "id_final": text,
                  "review_required": record.get("review_required", False), "note": record.get("note", "")}
    # Singleton QA uses a local index; the immutable global index was checked above.
    qa = run_subtitle_qa(
        [{**expected, "block_index": 1, "primary_text": expected["tr_text"]}],
        [{**normalized, "block_index": 1, "tr_final": expected["tr_text"]}],
        names_config=policy["names"], religious_config=policy["religious"],
        preferred_max_cps=policy["timing_qa"]["maximum_cps"],
        line_limit=policy["series"]["subtitle"]["qa_max_chars_per_line"],
    )
    if not qa["passed"] or normalized["review_required"]:
        return None, list(dict.fromkeys(
            [item["code"] for item in qa["issues"] if item.get("severity") != "warning"]
            or ["translator_review_required"]))
    return normalized, []


def _context(records, owned_uids, radius):
    owned = set(owned_uids)
    positions = [index for index, record in enumerate(records) if record["block_uid"] in owned]
    context = {index for position in positions
               for index in range(max(0, position - radius), min(len(records), position + radius + 1))
               if records[index]["block_uid"] not in owned}
    return [copy.deepcopy(records[index]) for index in sorted(context)]


def _worker_input(records, freeze, worker_id, owned_uids, radius):
    owned = set(owned_uids)
    return {"format": "mas-id-worker-input-1", "freeze": freeze, "worker_id": worker_id,
            "owned_uids": owned_uids,
            "records": [copy.deepcopy(record) for record in records if record["block_uid"] in owned],
            "read_only_context": _context(records, owned_uids, radius),
            "instructions": ID_TRANSLATION_INSTRUCTIONS + "\n" + WORKSPACE_INSTRUCTIONS}


def prepare_id_translation_workspaces(pack_path, output_dir, *, scope_uids=None, previous_result=None):
    schema, _, records, freeze = _pack(pack_path)
    policy = schema["production_policy"]
    ordered_uids = [record["block_uid"] for record in records]
    scope = ordered_uids if scope_uids is None else list(scope_uids)
    if not scope or scope != [uid for uid in ordered_uids if uid in set(scope)]:
        raise IDTranslationError("Workspace scope must be exact unique UIDs in schema order")
    expected = {record["block_uid"]: record for record in records}
    seed, reused, seed_reviews = [], [], []
    if previous_result is not None:
        old = copy.deepcopy(previous_result)
        claimed = old.pop("result_sha256", None)
        if claimed != digest(old) or old.get("format") != "mas-id-workspace-result-1":
            raise IDTranslationError("Previous workspace result binding changed")
        old_uids = [record["block_uid"] for record in old["accepted_records"]]
        if len(old_uids) != len(set(old_uids)) or old_uids != [uid for uid in old["scope_uids"] if uid in old_uids]:
            raise IDTranslationError("Previous workspace result has ambiguous UID ownership")
        if old["freeze"]["policy_sha256"] == freeze["policy_sha256"]:
            rejected = set(old["rejected_uids"])
            for prior in old["accepted_records"]:
                uid = prior["block_uid"]
                if uid not in scope or uid in rejected:
                    continue
                current = expected[uid]
                # Reuse text only as a draft; the new pack's immutable echo and QA are rebuilt.
                prior_immutable = {key: value for key, value in prior.items()
                                   if key not in ID_OUTPUT_FIELDS and key != "schema_sha256"}
                if prior_immutable != {key: value for key, value in current.items() if key != "schema_sha256"}:
                    continue
                candidate = {**current, **{key: prior[key] for key in ID_OUTPUT_FIELDS if key in prior}}
                checked, errors = _checked_record(current, candidate, policy)
                if not errors:
                    seed.append(checked)
                    reused.append(uid)
    seed_by_uid = {record["block_uid"]: record for record in seed}
    seed = [seed_by_uid[uid] for uid in scope if uid in seed_by_uid]
    if previous_result is not None and previous_result["freeze"]["policy_sha256"] == freeze["policy_sha256"]:
        current_items, _, _ = _review_state(records, seed_by_uid, scope, policy, [])
        current = {item["block_uid"]: item for item in current_items}
        bindings = ("source_record_sha256", "translation_sha256", "context_sha256", "policy_sha256")
        seed_reviews = [copy.deepcopy(review) for review in previous_result["reviews"]
                        if review["block_uid"] in current and all(review["checks"].values())
                        and all(review.get(key) == current[review["block_uid"]][key] for key in bindings)]
    pending = [uid for uid in scope if uid not in seed_by_uid]
    identity = {"freeze": freeze, "scope_uids": scope, "seed_records": seed, "seed_reviews": seed_reviews,
                "previous_result_sha256": None if previous_result is None else previous_result["result_sha256"]}
    destination = Path(output_dir) / digest(freeze) / digest(identity)
    workers = []
    for index in range(3):
        owned = pending[index * len(pending) // 3:(index + 1) * len(pending) // 3]
        worker_id = f"worker-{index + 1:02d}"
        value = _worker_input(records, freeze, worker_id, owned, policy["segmentation"]["context_blocks"])
        relative = f"{worker_id}/input.json"
        atomic_json(destination / relative, value)
        workers.append({"worker_id": worker_id, "input_file": relative,
                        "input_sha256": digest(value), "owned_uids": owned})
    workspace = {"format": "mas-id-translation-workspace-1", **identity,
                 "workers": workers, "reused_draft_uids": reused}
    atomic_json(destination / "workspace.json", {"payload": workspace, "sha256": digest(workspace)})
    return destination / "workspace.json"


def _risk_tags(record, policy, neighbors):
    source = record["tr_text"]
    tags = []
    if re.search(r"\d", source):
        tags.append("numbers_money")
    for name in policy["glossary"]["canonical_names"]:
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", source, re.IGNORECASE):
            tags.append("proper_names")
            break
    if any(term["source"].casefold() in source.casefold() for term in policy["glossary"]["religious_terms"]):
        tags.append("religious_meaning")
    if re.search(r"\b(bey|hanım|efendim|sayın|müdür|doktor)\b", source, re.IGNORECASE):
        tags.append("respectful_register")
    if source.rstrip().endswith(("...", "-")) or len(source.split()) >= 6 and len(record["id_final"].split()) <= 2:
        tags.append("meaning_compression_or_fragment")
    if record.get("note", "").strip():
        tags.append("translator_note")
    boundary = policy["segmentation"]["hard_silence_ms"]
    for neighbor in neighbors:
        if (neighbor["block_index"] == record["block_index"] - 1
                and record["start_ms"] - neighbor["end_ms"] >= boundary
                or neighbor["block_index"] == record["block_index"] + 1
                and neighbor["start_ms"] - record["end_ms"] >= boundary):
            tags.append("speech_gap_cue_ownership")
            break
    return tags


def _review_state(records, accepted, scope, policy, reviews):
    expected = {record["block_uid"]: record for record in records}
    risk_items = []
    for uid in scope:
        if uid not in accepted:
            continue
        context = _context(records, [uid], max(1, policy["segmentation"]["context_blocks"]))
        tags = _risk_tags(accepted[uid], policy, context)
        if tags:
            source_identity = {key: value for key, value in expected[uid].items() if key != "schema_sha256"}
            context_identity = [{key: value for key, value in item.items() if key != "schema_sha256"} for item in context]
            risk_items.append({"block_uid": uid, "risk_tags": tags, "source_record_sha256": digest(source_identity),
                               "translation_sha256": digest(accepted[uid]["id_final"]),
                               "context_sha256": digest(context_identity), "policy_sha256": policy["policy_sha256"],
                               "record": accepted[uid], "read_only_context": context,
                               "required_checks": list(REVIEW_CHECKS)})
    risk_by_uid = {item["block_uid"]: item for item in risk_items}
    approved, reviewed, rejected = set(), set(), {}
    for raw_review in reviews:
        review = _json_clone(raw_review)
        if not isinstance(review, dict):
            raise IDTranslationError("Meaning/register review must be an object")
        uid = review.get("block_uid")
        item = risk_by_uid.get(uid)
        if item is None or uid in reviewed:
            raise IDTranslationError("Review returned an unknown or duplicate cue")
        reviewed.add(uid)
        if (review.get("source_record_sha256") != item["source_record_sha256"]
                or review.get("translation_sha256") != item["translation_sha256"]
                or review.get("context_sha256") != item["context_sha256"]
                or review.get("policy_sha256") != item["policy_sha256"]
                or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                or review["reviewer"] in {"worker-01", "worker-02", "worker-03"}
                or not isinstance(review.get("checks"), dict) or set(review["checks"]) != set(REVIEW_CHECKS)
                or any(type(value) is not bool for value in review["checks"].values())):
            raise IDTranslationError("Meaning/register review binding is invalid")
        if all(review["checks"].values()):
            approved.add(uid)
        else:
            rejected[uid] = ["external_" + name for name, passed in review["checks"].items() if not passed]
    pending = [uid for uid in risk_by_uid if uid not in approved and uid not in rejected]
    return risk_items, pending, rejected


def collect_id_translation_workspaces(pack_path, workspace_path, worker_returns, *, reviews=(), out_zip=None):
    schema, manifest, records, freeze = _pack(pack_path)
    policy = schema["production_policy"]
    envelope = read_json(workspace_path)
    workspace = envelope.get("payload")
    if (not isinstance(workspace, dict) or envelope.get("sha256") != digest(workspace)
            or workspace.get("format") != "mas-id-translation-workspace-1" or workspace.get("freeze") != freeze):
        raise IDTranslationError("Workspace freeze binding changed")
    expected = {record["block_uid"]: record for record in records}
    scope = workspace["scope_uids"]
    if scope != [record["block_uid"] for record in records if record["block_uid"] in set(scope)]:
        raise IDTranslationError("Workspace scope binding changed")
    seed = workspace["seed_records"]
    seed_uids = [record["block_uid"] for record in seed]
    assigned = [uid for worker in workspace["workers"] for uid in worker["owned_uids"]]
    if (len(workspace["workers"]) != 3 or len(seed_uids) != len(set(seed_uids))
            or set(seed_uids) & set(assigned) or len(assigned) != len(set(assigned))
            or [uid for uid in scope if uid not in seed_uids] != assigned
            or any(uid not in scope for uid in seed_uids)):
        raise IDTranslationError("Workspace UID ownership changed")
    if set(worker_returns) - {worker["worker_id"] for worker in workspace["workers"]}:
        raise IDTranslationError("Unknown translation worker")
    accepted, rejected = {}, {}
    for record in seed:
        checked, errors = _checked_record(expected[record["block_uid"]], record, policy)
        if errors:
            raise IDTranslationError("Retained translation draft no longer passes QA")
        accepted[record["block_uid"]] = checked
    for index, worker in enumerate(workspace["workers"], start=1):
        worker_id = f"worker-{index:02d}"
        if worker["worker_id"] != worker_id or worker["input_file"] != f"{worker_id}/input.json":
            raise IDTranslationError("Unsafe translation worker input path")
        path = Path(workspace_path).parent / worker["input_file"]
        value = read_json(path)
        original = _worker_input(records, freeze, worker_id, worker["owned_uids"], policy["segmentation"]["context_blocks"])
        if value != original or worker["input_sha256"] != digest(value):
            raise IDTranslationError("Worker input or read-only context changed")
        returned = worker_returns.get(worker_id, [])
        if not isinstance(returned, list) or any(not isinstance(record, dict) for record in returned):
            raise IDTranslationError("Worker return must be a list of exact cue records")
        uids = [record.get("block_uid") for record in returned]
        if uids != [uid for uid in worker["owned_uids"] if uid in uids]:
            raise IDTranslationError("Worker returned duplicate, reordered, or unowned/context UIDs")
        for uid in worker["owned_uids"]:
            if uid not in uids:
                rejected[uid] = ["missing_translation"]
        for record in returned:
            uid = record["block_uid"]
            checked, errors = _checked_record(expected[uid], record, policy)
            if errors:
                rejected[uid] = errors
            else:
                accepted[uid] = checked
    reviews = list(workspace["seed_reviews"]) + list(reviews)
    risk_items, pending_review, review_rejections = _review_state(records, accepted, scope, policy, reviews)
    rejected.update(review_rejections)
    ready = not rejected and not pending_review
    ordered = [accepted[uid] for uid in scope if uid in accepted]
    result = {"format": "mas-id-workspace-result-1", "freeze": freeze, "scope_uids": scope,
              "status": "READY_SCOPE" if ready else "BLOCKED", "accepted_records": ordered,
              "rejected_uids": [uid for uid in scope if uid in rejected], "rejections": rejected,
              "review_pending_uids": pending_review, "risk_review_items": risk_items,
              "reviews": list(reviews), "semantic_acceptance": "EXTERNAL_REVIEW_ASSERTIONS_ONLY"}
    if out_zip is not None:
        if not ready or len(scope) != len(records):
            raise IDTranslationError("Cannot emit exact return ZIP before complete scope and quality review")
        validate_id_translation_records(schema, ordered)
        create_id_translation_output_zip(schema, ordered, out_zip, input_manifest=manifest)
        result["status"] = "EXACT_RETURN_CREATED"
        result["output_zip_sha256"] = hashlib.sha256(Path(out_zip).read_bytes()).hexdigest()
    result["result_sha256"] = digest(result)
    atomic_json(Path(workspace_path).parent / ("result-" + result["result_sha256"] + ".json"), result)
    if out_zip is not None:
        atomic_json(Path(str(out_zip) + ".workspace.json"), result)
    return result


def validate_id_workspace_output(pack_path, translated_zip, *, receipt_path=None):
    schema, manifest, records, freeze = _pack(pack_path)
    result = read_json(receipt_path or Path(str(translated_zip) + ".workspace.json"))
    payload = copy.deepcopy(result)
    claimed = payload.pop("result_sha256", None)
    if (claimed != digest(payload) or payload.get("format") != "mas-id-workspace-result-1"
            or payload.get("freeze") != freeze or payload.get("status") != "EXACT_RETURN_CREATED"
            or payload.get("output_zip_sha256") != hashlib.sha256(Path(translated_zip).read_bytes()).hexdigest()):
        raise IDTranslationError("Workspace output receipt binding changed")
    ordered = load_and_validate_id_translation_zip(schema, translated_zip, input_manifest=manifest).ordered_records(schema)
    scope = [record["block_uid"] for record in records]
    if (payload.get("scope_uids") != scope or not _json_equal(payload.get("accepted_records"), ordered)
            or payload.get("rejected_uids") != [] or payload.get("rejections") != {}
            or payload.get("review_pending_uids") != []):
        raise IDTranslationError("Workspace output does not preserve complete validated scope")
    for expected, record in zip(records, ordered):
        if _checked_record(expected, record, schema["production_policy"])[1]:
            raise IDTranslationError("Workspace output no longer passes cue QA")
    items, pending, rejected = _review_state(records, {record["block_uid"]: record for record in ordered},
                                            scope, schema["production_policy"], payload["reviews"])
    if pending or rejected or not _json_equal(items, payload.get("risk_review_items")):
        raise IDTranslationError("Workspace output lacks exact meaning/register review")
    return result
