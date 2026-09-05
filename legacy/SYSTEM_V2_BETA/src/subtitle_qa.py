"""Deterministic final subtitle quality assurance.

Structural translation validation must run before this module.  This layer
checks final subtitle timing, layout, reading speed, immutable anchors, names,
money, and mandatory religious-expression handling.  It reports problems; it
never rewrites dialogue or timing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
import re
import unicodedata
from typing import Any

try:  # Supports both ``import src.subtitle_qa`` and Colab's ``import subtitle_qa``.
    from .srt import SRTError, visible_length, wrap_text
except ImportError:  # pragma: no cover - exercised in Colab notebook mode
    from srt import SRTError, visible_length, wrap_text


class SubtitleQAError(RuntimeError):
    """Raised when final subtitles do not satisfy hard QA gates."""


DEFAULT_NAMES: tuple[str, ...] = (
    "Levent Bartıner",
    "Bartıner",
    "Defne",
    "Kadir",
    "Tolga",
    "Levent",
    "Mine",
    "Melis",
    "Özlem",
    "Selim",
    "Selma",
    "Sultan",
    "Zeynep",
    "Zeyno",
    "Suzi",
    "Leyla",
    "Oğuz",
    "Yavuz",
    "Zeliha",
    "Emindağ",
)

DEFAULT_NAME_ALIASES: Mapping[str, tuple[str, ...]] = {
    "Bartıner": ("Bartiner", "Bartıner"),
    "Levent Bartıner": ("Levent Bartiner", "Levent Bartıner"),
    "Emindağ": ("Emindag", "Emin Dağ", "Emindağ"),
    "Oğuz": ("Oguz", "Oğuz"),
    "Özlem": ("Ozlem", "Özlem"),
}

DEFAULT_RELIGIOUS_TERMS: Mapping[str, tuple[str, ...]] = {
    "Allah aşkına": ("Demi Allah",),
    "Allah Allah": ("Ya Allah",),
    "Allah'ım": ("Ya Allah",),
    "Ya Rabbim": ("Ya Rabb",),
    "İnşallah": ("Insyaallah", "Insya Allah"),
    "Maşallah": ("Masyaallah", "Masya Allah"),
    "Estağfurullah": ("Astagfirullah",),
    "Tövbe estağfurullah": ("Tobat, astagfirullah", "Tobat astagfirullah"),
    "La havle vela kuvvete illa billah": (
        "La hawla wala quwwata illa billah",
    ),
}

_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d+)?(?![\w])",
    flags=re.UNICODE,
)
_MONEY_RE = re.compile(
    r"(?ix)(?:"
    r"(?:₺|\$|€|£|\b(?:try|tl|usd|eur|gbp|idr|rp|lira|dolar|euro)\b)\s*"
    r"(?P<before>[-+]?\d[\d.,\s]*)"
    r"|(?P<after>[-+]?\d[\d.,\s]*)\s*"
    r"(?:₺|\$|€|£|\b(?:try|tl|usd|eur|gbp|idr|rp|lira|dolar|euro)\b)"
    r")"
)
_DIALOGUE_RE = re.compile(r"^\s*[-–—]\s*\S")


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return (
        text.casefold()
        .replace("’", "'")
        .replace("`", "'")
        .replace("ı", "i")
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalized(text)
    normalized_phrase = _normalized(phrase)
    pattern = r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)"
    return re.search(pattern, normalized_text, flags=re.UNICODE) is not None


def _contains_canonical_spelling(text: str, value: str) -> bool:
    """Case-insensitive match which retains Turkish spelling distinctions."""

    def canonical_casefold(raw: str) -> str:
        folded = unicodedata.normalize("NFC", raw).casefold()
        # Python's locale-neutral casefold maps Turkish capital İ to i + a
        # combining dot. Removing only that dot accepts normal capitalization
        # without collapsing ı/i, ö/o, ğ/g or ş/s.
        return unicodedata.normalize("NFC", folded.replace("i\u0307", "i"))

    normalized_text = canonical_casefold(text)
    normalized_value = canonical_casefold(value)
    return (
        re.search(
            r"(?<!\w)" + re.escape(normalized_value) + r"(?!\w)",
            normalized_text,
            flags=re.UNICODE,
        )
        is not None
    )


def _contains_allah(text: str) -> bool:
    # Includes the common apostrophe-less ASR spelling "Allahım", while not
    # treating the suffix in "billah" as a standalone Allah reference.
    return (
        re.search(
            r"(?<!\w)allah(?:'?im)?(?!\w)",
            _normalized(text),
            flags=re.UNICODE,
        )
        is not None
    )


def _is_music_marker(text: str) -> bool:
    normalized = _normalized(text).strip()
    normalized = normalized.strip("[](){} ")
    return normalized in {"muzik", "music", "musik", "♪", "♫"}


def _canonical_number(raw: str) -> str:
    value = raw.strip().replace(" ", "")
    sign = ""
    if value[:1] in {"+", "-"}:
        sign, value = value[0], value[1:]
    if not value:
        return sign
    separators = [position for position, char in enumerate(value) if char in ".,"]
    if not separators:
        integer, fraction = value, ""
    else:
        last = separators[-1]
        digits_after = len(value) - last - 1
        separator_count = len(separators)
        # A single 1- or 2-digit suffix is treated as a decimal. Repeated
        # separators and 3-digit groups are treated as thousands formatting.
        decimal = digits_after in {1, 2} and (
            separator_count == 1 or value[last] != value[separators[-2]]
        )
        if decimal:
            integer = re.sub(r"[.,]", "", value[:last])
            fraction = re.sub(r"[.,]", "", value[last + 1 :]).rstrip("0")
        else:
            integer = re.sub(r"[.,]", "", value)
            fraction = ""
    integer = integer.lstrip("0") or "0"
    result = sign + integer
    if fraction:
        result += "." + fraction
    return result


def extract_number_anchors(text: str) -> Counter[str]:
    """Extract normalized numeric anchors, retaining repeated values."""

    return Counter(_canonical_number(match.group(0)) for match in _NUMBER_RE.finditer(text))


def extract_money_anchors(text: str) -> Counter[str]:
    """Extract normalized monetary numeric values independent of currency wording."""

    anchors: Counter[str] = Counter()
    for match in _MONEY_RE.finditer(text):
        raw = match.group("before") or match.group("after") or ""
        number = _NUMBER_RE.search(raw.strip())
        if number:
            anchors[_canonical_number(number.group(0))] += 1
    return anchors


def _names_from_config(config: Any) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {
        name: {name, *DEFAULT_NAME_ALIASES.get(name, ())} for name in DEFAULT_NAMES
    }
    if config is None:
        return names

    def add(canonical: Any, aliases: Any = ()) -> None:
        if not isinstance(canonical, str) or not canonical.strip():
            return
        canonical = canonical.strip()
        bucket = names.setdefault(canonical, {canonical})
        if isinstance(aliases, str):
            bucket.add(aliases.strip())
        elif isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
            bucket.update(str(alias).strip() for alias in aliases if str(alias).strip())

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        for item in config:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, Mapping):
                add(
                    item.get("canonical") or item.get("name") or item.get("value"),
                    item.get("aliases", ()),
                )
    elif isinstance(config, Mapping):
        candidate_list = (
            config.get("names")
            or config.get("canonical_names")
            or config.get("known_names")
        )
        if candidate_list is not None:
            nested = _names_from_config(candidate_list)
            for canonical, aliases in nested.items():
                names.setdefault(canonical, {canonical}).update(aliases)
        aliases_map = (
            config.get("aliases")
            or config.get("name_aliases")
            or config.get("forbidden_variants")
        )
        if isinstance(aliases_map, Mapping):
            for key, value in aliases_map.items():
                # Accept both canonical -> aliases and alias -> canonical.
                if isinstance(value, str) and value in names and key not in names:
                    add(value, (str(key),))
                else:
                    add(str(key), value)
        for key, value in config.items():
            if key in {
                "names",
                "canonical_names",
                "known_names",
                "aliases",
                "name_aliases",
                "forbidden_variants",
            }:
                continue
            if isinstance(value, (list, tuple, set)):
                add(str(key), value)
    return names


def _religious_from_config(config: Any) -> dict[str, tuple[str, ...]]:
    mappings: dict[str, tuple[str, ...]] = dict(DEFAULT_RELIGIOUS_TERMS)
    if config is None:
        return mappings

    def add(turkish: Any, indonesian: Any) -> None:
        if not isinstance(turkish, str) or not turkish.strip():
            return
        if isinstance(indonesian, str):
            accepted = (indonesian.strip(),)
        elif isinstance(indonesian, Sequence) and not isinstance(indonesian, (str, bytes)):
            accepted = tuple(str(item).strip() for item in indonesian if str(item).strip())
        else:
            return
        if accepted:
            mappings[turkish.strip()] = accepted

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        for item in config:
            if isinstance(item, Mapping):
                add(
                    item.get("turkish") or item.get("source") or item.get("tr"),
                    item.get("indonesian")
                    or item.get("preferred_indonesian")
                    or item.get("target")
                    or item.get("id")
                    or item.get("accepted"),
                )
    elif isinstance(config, Mapping):
        nested = config.get("terms") or config.get("mappings")
        if nested is not None and nested is not config:
            mappings.update(_religious_from_config(nested))
        for source, target in config.items():
            if source in {"terms", "mappings", "allah_preservation"}:
                continue
            if isinstance(target, Mapping):
                add(
                    target.get("turkish") or target.get("source") or source,
                    target.get("indonesian")
                    or target.get("preferred_indonesian")
                    or target.get("target")
                    or target.get("accepted"),
                )
            else:
                add(source, target)
    return mappings


def _issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    block: Mapping[str, Any] | None,
    message: str,
    severity: str = "error",
    language: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> None:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "block_uid": block.get("block_uid") if block else None,
        "block_index": block.get("block_index") if block else None,
        "message": message,
    }
    if language:
        issue["language"] = language
    if expected is not None:
        issue["expected"] = expected
    if actual is not None:
        issue["actual"] = actual
    issues.append(issue)


def _translation_records(
    value: Any,
) -> tuple[list[Mapping[str, Any]], int, Any | None]:
    """Return records, duplicate count, and an optional upstream report."""

    upstream_report = None
    if hasattr(value, "records_by_uid"):
        upstream_report = getattr(value, "report", value)
        value = getattr(value, "records_by_uid")
    elif hasattr(value, "validated_by_uid"):
        upstream_report = getattr(value, "report", value)
        value = getattr(value, "validated_by_uid")
    if isinstance(value, Mapping):
        records: list[Mapping[str, Any]] = []
        for key, record in value.items():
            if not isinstance(record, Mapping):
                record = {"block_uid": key, "_invalid_record": record}
            elif "block_uid" not in record:
                record = dict(record)
                record["block_uid"] = key
            records.append(record)
        return records, 0, upstream_report
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = [record for record in value if isinstance(record, Mapping)]
        counts = Counter(str(record.get("block_uid", "")) for record in records)
        duplicate_count = sum(count - 1 for uid, count in counts.items() if uid and count > 1)
        return records, duplicate_count, upstream_report
    raise TypeError("translations_by_uid must be a mapping or sequence of records")


def _upstream_count(report: Any, *names: str) -> int:
    if report is None:
        return 0
    source: Any = report
    if hasattr(report, "to_dict"):
        try:
            source = report.to_dict()
        except Exception:
            source = report
    for name in names:
        if isinstance(source, Mapping) and name in source:
            try:
                return int(source[name])
            except (TypeError, ValueError):
                return 0
        if hasattr(report, name):
            try:
                return int(getattr(report, name))
            except (TypeError, ValueError):
                return 0
    return 0


def _upstream_passed(report: Any) -> bool:
    if report is None:
        return False
    source: Any = report
    if hasattr(report, "to_dict"):
        try:
            source = report.to_dict()
        except Exception:
            source = report
    if isinstance(source, Mapping):
        return source.get("ok") is True
    return getattr(report, "ok", False) is True


def _review_is_justified(record: Mapping[str, Any]) -> bool:
    note = record.get("note")
    return bool(
        record.get("review_required") is True
        and isinstance(note, str)
        and note.strip()
    )


def _vad_number(vad_info: Any, keys: Sequence[str]) -> float | None:
    if not isinstance(vad_info, Mapping):
        return None
    for key in keys:
        value = vad_info.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _has_risk(block: Mapping[str, Any], *needles: str) -> bool:
    raw = block.get("risk_flags", ())
    if isinstance(raw, str):
        flags = (raw,)
    elif isinstance(raw, Sequence):
        flags = tuple(str(item) for item in raw)
    else:
        flags = ()
    normalized_flags = {_normalized(flag).replace(" ", "_") for flag in flags}
    return any(_normalized(needle).replace(" ", "_") in normalized_flags for needle in needles)


def _layout_for_qa(text: str, line_limit: int) -> tuple[list[str], str | None]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")
    if len(raw_lines) > 2:
        return raw_lines, "more_than_two_lines"
    try:
        laid_out = wrap_text(normalized, target=42, max_lines=2, hard_limit=line_limit)
    except SRTError:
        return raw_lines, "layout_error"
    return laid_out.split("\n"), None


def run_subtitle_qa(
    blocks: Sequence[Mapping[str, Any]],
    translations_by_uid: Any,
    names_config: Any = None,
    religious_config: Any = None,
    preferred_max_cps: float = 20,
    line_limit: int = 84,
) -> dict[str, Any]:
    """Run final deterministic QA and return a JSON-serializable report."""

    if preferred_max_cps <= 0:
        raise ValueError("preferred_max_cps must be greater than zero")
    if line_limit < 1:
        raise ValueError("line_limit must be positive")

    issues: list[dict[str, Any]] = []
    schema_uids: list[str] = []
    block_by_uid: dict[str, Mapping[str, Any]] = {}
    duplicate_schema_uid_count = 0
    for position, block in enumerate(blocks, start=1):
        uid = block.get("block_uid")
        if not isinstance(uid, str) or not uid:
            uid = f"<invalid-schema-position-{position}>"
            _issue(
                issues,
                code="invalid_schema_uid",
                block=block,
                message=f"Schema block at position {position} has no valid block_uid",
            )
        if uid in block_by_uid:
            duplicate_schema_uid_count += 1
        else:
            block_by_uid[uid] = block
        schema_uids.append(uid)

    records, duplicate_translation_count, upstream_report = _translation_records(
        translations_by_uid
    )
    upstream_passed = _upstream_passed(upstream_report)
    record_uid_list = [str(record.get("block_uid", "")) for record in records]
    record_counts = Counter(record_uid_list)
    duplicate_translation_count += _upstream_count(
        upstream_report, "duplicate_translation_count", "duplicate_uid_count"
    )
    record_by_uid: dict[str, Mapping[str, Any]] = {}
    for record in records:
        uid = str(record.get("block_uid", ""))
        if uid and uid not in record_by_uid:
            record_by_uid[uid] = record

    schema_uid_set = set(schema_uids)
    output_uid_set = {uid for uid in record_uid_list if uid}
    missing_uids = [uid for uid in schema_uids if uid not in output_uid_set]
    extra_uids = sorted(output_uid_set - schema_uid_set)
    missing_translation_count = len(missing_uids) + _upstream_count(
        upstream_report, "missing_translation_count", "missing_uid_count"
    )
    extra_translation_count = len(extra_uids) + _upstream_count(
        upstream_report, "extra_translation_count", "extra_uid_count"
    )
    uid_mismatch_count = duplicate_schema_uid_count + _upstream_count(
        upstream_report, "uid_mismatch_count"
    )
    for uid in missing_uids:
        _issue(
            issues,
            code="missing_translation_uid",
            block=block_by_uid.get(uid),
            message=f"No translation record exists for {uid}",
            expected=uid,
            actual=None,
        )
    for uid in extra_uids:
        _issue(
            issues,
            code="extra_translation_uid",
            block=None,
            message=f"Translation output contains unknown UID {uid}",
            expected=None,
            actual=uid,
        )

    order_mismatch_count = 0
    if len(record_uid_list) == len(schema_uids) and record_uid_list != schema_uids:
        order_mismatch_count = sum(
            actual != expected
            for actual, expected in zip(record_uid_list, schema_uids)
        )
        _issue(
            issues,
            code="translation_order_mismatch",
            block=None,
            message="Translation record order differs from immutable schema order",
            expected=schema_uids[:10],
            actual=record_uid_list[:10],
        )
    order_mismatch_count += _upstream_count(upstream_report, "order_mismatch_count")

    # Every record must carry one identical non-empty schema hash. If schema
    # blocks echo it, that authoritative value wins.
    expected_hashes = {
        str(block.get("schema_sha256"))
        for block in blocks
        if isinstance(block.get("schema_sha256"), str) and block.get("schema_sha256")
    }
    expected_hash = next(iter(expected_hashes)) if len(expected_hashes) == 1 else None
    record_hashes = [record.get("schema_sha256") for record in records]
    if expected_hash is None:
        nonempty_hashes = [value for value in record_hashes if isinstance(value, str) and value]
        expected_hash = nonempty_hashes[0] if nonempty_hashes else None
    schema_mismatch_count = 0
    if len(expected_hashes) > 1:
        schema_mismatch_count += len(expected_hashes) - 1
    for record, actual_hash in zip(records, record_hashes):
        if not isinstance(actual_hash, str) or not actual_hash or (
            expected_hash is not None and actual_hash != expected_hash
        ):
            schema_mismatch_count += 1
            _issue(
                issues,
                code="schema_hash_mismatch",
                block=block_by_uid.get(str(record.get("block_uid", ""))),
                message="Translation schema_sha256 is missing or inconsistent",
                expected=expected_hash,
                actual=actual_hash,
            )
    schema_mismatch_count += _upstream_count(
        upstream_report, "schema_mismatch_count", "schema_hash_mismatch_count"
    )

    # Mapping-key versus echoed UID mismatches may be surfaced by an upstream
    # validator. A raw sequence has no independent key to compare.
    if isinstance(translations_by_uid, Mapping):
        for key, record in translations_by_uid.items():
            if isinstance(record, Mapping) and record.get("block_uid", key) != key:
                uid_mismatch_count += 1
                _issue(
                    issues,
                    code="mapping_uid_mismatch",
                    block=block_by_uid.get(str(key)),
                    message="Translation mapping key and record block_uid differ",
                    expected=key,
                    actual=record.get("block_uid"),
                )

    names = _names_from_config(names_config)
    religious_terms = _religious_from_config(religious_config)
    counts: Counter[str] = Counter()
    anchor_failure_blocks: set[str] = set()
    tr_block_count = 0
    id_block_count = 0

    for position, block in enumerate(blocks, start=1):
        uid = str(block.get("block_uid", ""))
        record = record_by_uid.get(uid)
        start_ms = block.get("start_ms")
        end_ms = block.get("end_ms")
        valid_timing = (
            isinstance(start_ms, int)
            and not isinstance(start_ms, bool)
            and isinstance(end_ms, int)
            and not isinstance(end_ms, bool)
            and start_ms >= 0
            and end_ms > start_ms
        )
        if not valid_timing:
            counts["timing_error_count"] += 1
            _issue(
                issues,
                code="invalid_timing",
                block=block,
                message="Block timing must be non-negative integer milliseconds with end > start",
                expected="0 <= start_ms < end_ms",
                actual={"start_ms": start_ms, "end_ms": end_ms},
            )

        block_index = block.get("block_index")
        if block_index != position:
            uid_mismatch_count += 1
            _issue(
                issues,
                code="block_index_mismatch",
                block=block,
                message="Immutable block_index is not contiguous",
                expected=position,
                actual=block_index,
            )

        if record is None:
            continue
        tr_text = record.get("tr_final")
        id_text = record.get("id_final")
        if isinstance(tr_text, str):
            tr_block_count += 1
        if isinstance(id_text, str):
            id_block_count += 1

        for language, text in (("tr", tr_text), ("id", id_text)):
            if not isinstance(text, str) or not text.strip():
                counts["empty_text_count"] += 1
                _issue(
                    issues,
                    code="empty_text",
                    block=block,
                    language=language,
                    message=f"Final {language.upper()} subtitle text is empty",
                )
                continue
            try:
                text.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                counts["utf8_error_count"] += 1
                _issue(
                    issues,
                    code="invalid_utf8_text",
                    block=block,
                    language=language,
                    message="Subtitle contains a Unicode surrogate and cannot be encoded as UTF-8",
                )
            lines, layout_error = _layout_for_qa(text, line_limit)
            if layout_error == "more_than_two_lines":
                counts["more_than_two_lines_count"] += 1
                counts["srt_structure_error_count"] += 1
                _issue(
                    issues,
                    code="more_than_two_lines",
                    block=block,
                    language=language,
                    message=f"Subtitle has {len(lines)} visible lines; maximum is 2",
                    expected=2,
                    actual=len(lines),
                )
            elif layout_error == "layout_error":
                counts["srt_structure_error_count"] += 1
                # If the layout cannot fit two lines, at least one resulting
                # line necessarily exceeds the configured hard limit.
                counts["line_over_84_count"] += 1
                _issue(
                    issues,
                    code="subtitle_layout_unrepresentable",
                    block=block,
                    language=language,
                    message=f"Subtitle cannot fit within 2 lines of {line_limit} characters",
                )
            long_lines = [visible_length(line) for line in lines if visible_length(line) > line_limit]
            if long_lines and layout_error != "layout_error":
                counts["line_over_84_count"] += len(long_lines)
                _issue(
                    issues,
                    code="line_over_limit",
                    block=block,
                    language=language,
                    message=f"Subtitle line exceeds {line_limit} visible characters",
                    expected=f"<= {line_limit}",
                    actual=max(long_lines),
                )
            if valid_timing:
                duration_seconds = (end_ms - start_ms) / 1_000
                characters = visible_length(text.replace("\n", " "))
                cps = characters / duration_seconds
                if cps > preferred_max_cps:
                    counts["high_cps_count"] += 1
                    _issue(
                        issues,
                        code="high_cps",
                        block=block,
                        language=language,
                        severity="warning",
                        message=(
                            f"Reading speed is {cps:.1f} CPS; preferred maximum is "
                            f"{preferred_max_cps:.1f} CPS"
                        ),
                        expected=f"<= {preferred_max_cps:.1f}",
                        actual=round(cps, 2),
                    )

        if not isinstance(tr_text, str) or not isinstance(id_text, str):
            continue

        # Targeted verification can include words from the neighbouring timing
        # window.  It is useful translation evidence, but it cannot make a
        # name mandatory in this immutable subtitle block.  Bind hard name
        # preservation only to same-block primary/YouTube evidence (or to the
        # corrected Turkish text itself).
        source_evidence = " ".join(
            str(block.get(field) or "")
            for field in ("primary_text", "youtube_text")
        )
        # The upstream translation validator may accept a source-name
        # correction only when TR/ID agree, forbidden spellings are absent,
        # and the record carries an explicit review note.  Do not re-impose
        # the rejected raw-ASR name in final QA after that trusted decision.
        # Raw records and unreviewed records retain the independent source
        # anchor check below.
        source_name_override = upstream_passed and _review_is_justified(record)
        source_for_names = tr_text if source_name_override else source_evidence + " " + tr_text
        name_failures: list[str] = []
        for canonical, aliases in names.items():
            mentioned = any(_contains_phrase(source_for_names, alias) for alias in aliases)
            if not mentioned:
                continue
            tr_ok = _contains_canonical_spelling(tr_text, canonical)
            id_ok = _contains_canonical_spelling(id_text, canonical)
            forbidden_present = sorted(
                alias
                for alias in aliases
                if alias != canonical
                and (
                    _contains_canonical_spelling(tr_text, alias)
                    or _contains_canonical_spelling(id_text, alias)
                )
            )
            if not tr_ok or not id_ok or forbidden_present:
                name_failures.append(canonical)
                _issue(
                    issues,
                    code="special_name_mismatch",
                    block=block,
                    message=f"Special name must use canonical spelling {canonical!r} in both languages",
                    expected=canonical,
                    actual={
                        "tr_final": tr_text,
                        "id_final": id_text,
                        "forbidden_variants": forbidden_present,
                    },
                )
        if name_failures:
            counts["special_name_mismatch_count"] += 1
            anchor_failure_blocks.add(uid)

        tr_numbers = extract_number_anchors(tr_text)
        id_numbers = extract_number_anchors(id_text)
        if tr_numbers != id_numbers:
            counts["numeric_mismatch_count"] += 1
            anchor_failure_blocks.add(uid)
            _issue(
                issues,
                code="numeric_anchor_mismatch",
                block=block,
                message="Numeric values or repetitions differ between Turkish and Indonesian",
                expected=dict(tr_numbers),
                actual=dict(id_numbers),
            )
        tr_money = extract_money_anchors(tr_text)
        id_number_pool = extract_number_anchors(id_text)
        if any(id_number_pool[value] < count for value, count in tr_money.items()):
            counts["money_mismatch_count"] += 1
            anchor_failure_blocks.add(uid)
            _issue(
                issues,
                code="money_value_mismatch",
                block=block,
                message="One or more Turkish money values are absent from Indonesian",
                expected=dict(tr_money),
                actual=dict(id_number_pool),
            )

        religious_failed = False
        for turkish_phrase, accepted_indonesian in religious_terms.items():
            if not _contains_phrase(tr_text, turkish_phrase):
                continue
            if not any(_contains_phrase(id_text, phrase) for phrase in accepted_indonesian):
                religious_failed = True
                _issue(
                    issues,
                    code="religious_expression_mismatch",
                    block=block,
                    message=f"Mandatory mapping for {turkish_phrase!r} is missing",
                    expected=list(accepted_indonesian),
                    actual=id_text,
                )
        if religious_failed:
            counts["religious_expression_mismatch_count"] += 1
            anchor_failure_blocks.add(uid)

        if _contains_allah(tr_text) and not _contains_allah(id_text):
            counts["allah_preservation_failure_count"] += 1
            anchor_failure_blocks.add(uid)
            only_semoga = _contains_phrase(id_text, "Semoga")
            _issue(
                issues,
                code="allah_replaced_by_semoga" if only_semoga else "allah_missing",
                block=block,
                message=(
                    "Turkish contains Allah, but Indonesian does not preserve Allah"
                    + (" and uses Semoga alone" if only_semoga else "")
                ),
                expected="Allah",
                actual=id_text,
            )

        evidence_parts = [
            str(block.get(field) or "").strip()
            for field in ("timing_text", "primary_text", "verification_text", "youtube_text")
            if str(block.get(field) or "").strip()
        ]
        evidence_only_music = bool(evidence_parts) and all(
            _is_music_marker(part) for part in evidence_parts
        )
        evidence_has_speech = any(
            not _is_music_marker(part)
            and bool(re.search(r"\w", part, flags=re.UNICODE))
            for part in evidence_parts
        )
        music_error = (
            (evidence_only_music and not _is_music_marker(tr_text))
            or (evidence_has_speech and _is_music_marker(tr_text))
        )
        if music_error:
            counts["music_speech_handling_count"] += 1
            _issue(
                issues,
                code="music_speech_handling",
                block=block,
                message=(
                    "Final Turkish text invents speech over music"
                    if evidence_only_music
                    else "Final Turkish text replaces supported speech with a music marker"
                ),
                expected=evidence_parts,
                actual=tr_text,
            )

        tr_lines = tr_text.replace("\r", "").split("\n")
        if sum(bool(_DIALOGUE_RE.match(line)) for line in tr_lines) > 1:
            counts["mixed_speaker_block_count"] += 1
            _issue(
                issues,
                code="mixed_speaker_block",
                block=block,
                message="Multiple dash-prefixed dialogue turns appear in one immutable block",
            )

        if record.get("review_required") is True:
            counts["review_required_count"] += 1
            _issue(
                issues,
                code="translator_review_required",
                block=block,
                severity="warning",
                message=str(record.get("note") or "Translator requested manual review"),
            )

    # Schema timing checks are performed once per block, independent of language.
    overlap_count = 0
    early_start_count = 0
    early_end_count = 0
    unresolved_internal_gap_count = 0
    mixed_risk_uids: set[str] = set()
    previous: Mapping[str, Any] | None = None
    for block in blocks:
        uid = str(block.get("block_uid", ""))
        start_ms = block.get("start_ms")
        end_ms = block.get("end_ms")
        if (
            previous is not None
            and isinstance(previous.get("end_ms"), int)
            and isinstance(start_ms, int)
            and start_ms < previous["end_ms"]
        ):
            overlap_count += 1
            _issue(
                issues,
                code="subtitle_overlap",
                block=block,
                message=f"Block overlaps previous UID {previous.get('block_uid')}",
                expected=f">= {previous.get('end_ms')}",
                actual=start_ms,
            )
        previous = block

        vad = block.get("vad_info")
        speech_start = _vad_number(
            vad,
            ("first_word_start_ms", "speech_start_ms", "vad_start_ms", "start_ms"),
        )
        speech_end = _vad_number(
            vad,
            ("last_word_end_ms", "speech_end_ms", "vad_end_ms", "end_ms"),
        )
        if isinstance(start_ms, int) and speech_start is not None and start_ms < speech_start - 80:
            early_start_count += 1
            _issue(
                issues,
                code="early_subtitle_start",
                block=block,
                message="Subtitle starts more than 80 ms before reliable speech",
                expected=f">= {int(speech_start - 80)}",
                actual=start_ms,
            )
        if isinstance(end_ms, int) and speech_end is not None and end_ms < speech_end:
            early_end_count += 1
            _issue(
                issues,
                code="early_subtitle_end",
                block=block,
                message="Subtitle disappears before the last reliable spoken word ends",
                expected=f">= {int(speech_end)}",
                actual=end_ms,
            )
        maximum_gap = _vad_number(
            vad,
            ("max_internal_gap_ms", "internal_silence_ms", "largest_internal_gap_ms"),
        )
        gap_resolved = isinstance(vad, Mapping) and vad.get("internal_gap_resolved") is True
        if (
            (maximum_gap is not None and maximum_gap > 1_000 and not gap_resolved)
            or _has_risk(block, "unresolved_internal_gap", "internal_gap_over_1s")
        ):
            unresolved_internal_gap_count += 1
            _issue(
                issues,
                code="unresolved_internal_gap",
                block=block,
                message="Internal speech gap over 1,000 ms remains unresolved",
                actual=maximum_gap,
            )
        if _has_risk(block, "mixed_speaker", "mixed_speaker_block"):
            mixed_risk_uids.add(uid)

    # Avoid double-counting a block found both from text and a segmentation flag.
    text_mixed_uids = {
        str(issue.get("block_uid"))
        for issue in issues
        if issue.get("code") == "mixed_speaker_block"
    }
    new_mixed_uids = mixed_risk_uids - text_mixed_uids
    counts["mixed_speaker_block_count"] += len(new_mixed_uids)
    for uid in sorted(new_mixed_uids):
        _issue(
            issues,
            code="mixed_speaker_block",
            block=block_by_uid.get(uid),
            message="Segmentation diagnostics flag more than one speaker in this block",
        )

    positional_translation_mismatch_count = _upstream_count(
        upstream_report,
        "positional_translation_mismatch_count",
        "positional_mismatch_count",
    )
    # Validators may annotate individual records while retaining a mapping.
    positional_translation_mismatch_count += sum(
        record.get("positional_alignment_ok") is False
        or record.get("alignment_ok") is False
        for record in records
    )

    report: dict[str, Any] = {
        "tr_block_count": tr_block_count,
        "id_block_count": id_block_count,
        "timings_identical": True,
        "timing_mismatch_count": 0,
        "missing_translation_count": missing_translation_count,
        "duplicate_translation_count": duplicate_translation_count,
        "extra_translation_count": extra_translation_count,
        "positional_translation_mismatch_count": positional_translation_mismatch_count,
        "overlap_count": overlap_count,
        "early_start_count": early_start_count,
        "early_end_count": early_end_count,
        "empty_text_count": counts["empty_text_count"],
        "more_than_two_lines_count": counts["more_than_two_lines_count"],
        "line_over_84_count": counts["line_over_84_count"],
        "mixed_speaker_block_count": counts["mixed_speaker_block_count"],
        "unresolved_internal_gap_count": unresolved_internal_gap_count,
        "schema_mismatch_count": schema_mismatch_count,
        "uid_mismatch_count": uid_mismatch_count,
        "order_mismatch_count": order_mismatch_count,
        "timing_error_count": counts["timing_error_count"],
        "special_name_mismatch_count": counts["special_name_mismatch_count"],
        "numeric_mismatch_count": counts["numeric_mismatch_count"],
        "money_mismatch_count": counts["money_mismatch_count"],
        "religious_expression_mismatch_count": counts[
            "religious_expression_mismatch_count"
        ],
        "allah_preservation_failure_count": counts[
            "allah_preservation_failure_count"
        ],
        "music_speech_handling_count": counts["music_speech_handling_count"],
        "anchor_mismatch_count": len(anchor_failure_blocks),
        "utf8_error_count": counts["utf8_error_count"],
        "srt_structure_error_count": counts["srt_structure_error_count"],
        "high_cps_count": counts["high_cps_count"],
        "review_required_count": counts["review_required_count"],
        "issues": issues,
    }
    hard_failures = _hard_failure_messages(report)
    report["passed"] = not hard_failures
    report["hard_failure_messages"] = hard_failures
    return report


_ZERO_REQUIRED: tuple[str, ...] = (
    "missing_translation_count",
    "duplicate_translation_count",
    "extra_translation_count",
    "positional_translation_mismatch_count",
    "overlap_count",
    "early_start_count",
    "early_end_count",
    "empty_text_count",
    "more_than_two_lines_count",
    "line_over_84_count",
    "mixed_speaker_block_count",
    "unresolved_internal_gap_count",
    "schema_mismatch_count",
    "uid_mismatch_count",
    "order_mismatch_count",
    "timing_mismatch_count",
    "timing_error_count",
    "special_name_mismatch_count",
    "numeric_mismatch_count",
    "money_mismatch_count",
    "religious_expression_mismatch_count",
    "allah_preservation_failure_count",
    "music_speech_handling_count",
    "utf8_error_count",
    "srt_structure_error_count",
)


def _hard_failure_messages(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if report.get("tr_block_count") != report.get("id_block_count"):
        failures.append(
            "tr_block_count != id_block_count "
            f"({report.get('tr_block_count')} != {report.get('id_block_count')})"
        )
    if report.get("timings_identical") is not True:
        failures.append("Turkish and Indonesian timings are not identical")
    for field in _ZERO_REQUIRED:
        value = report.get(field, 0)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            failures.append(f"{field} is not an integer: {value!r}")
            continue
        if numeric != 0:
            failures.append(f"{field}={numeric} (required 0)")
    return failures


def assert_final_qa(report: Mapping[str, Any]) -> None:
    """Raise a concise error unless every hard finalization gate passes."""

    failures = _hard_failure_messages(report)
    if not failures:
        return
    issue_examples: list[str] = []
    raw_issues = report.get("issues", ())
    if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes)):
        for issue in raw_issues:
            if not isinstance(issue, Mapping) or issue.get("severity") == "warning":
                continue
            uid = issue.get("block_uid") or "episode"
            issue_examples.append(f"{uid}: {issue.get('code')} - {issue.get('message')}")
            if len(issue_examples) == 5:
                break
    message = "Final subtitle QA failed:\n- " + "\n- ".join(failures)
    if issue_examples:
        message += "\nExamples:\n- " + "\n- ".join(issue_examples)
    raise SubtitleQAError(message)


__all__ = [
    "DEFAULT_NAMES",
    "DEFAULT_RELIGIOUS_TERMS",
    "SubtitleQAError",
    "assert_final_qa",
    "extract_money_anchors",
    "extract_number_anchors",
    "run_subtitle_qa",
]

