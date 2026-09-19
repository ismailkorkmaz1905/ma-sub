"""Strict structural and semantic validation for Work Ultra translations.

The only association key in this module is ``block_uid``.  Raw records remain
in a list until duplicate, missing, extra and ordering checks have completed;
only then is a UID map exposed to FINALIZE.
"""

from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

try:
    from .batches import (
        BatchError,
        BatchFile,
        build_translation_batches,
        validate_translation_pack,
    )
    from .schema import SchemaError, validate_episode_schema
except ImportError:  # Direct import after adding src/ to sys.path.
    from batches import (  # type: ignore
        BatchError,
        BatchFile,
        build_translation_batches,
        validate_translation_pack,
    )
    from schema import SchemaError, validate_episode_schema  # type: ignore


TRANSLATION_RECORD_FIELDS = frozenset(
    {
        "block_uid",
        "schema_sha256",
        "tr_final",
        "id_final",
        "review_required",
        "note",
    }
)

TRANSLATION_REPORT_FIELDS = frozenset(
    {
        "schema_sha256",
        "total_input_blocks",
        "total_output_blocks",
        "missing_block_count",
        "duplicate_block_count",
        "review_required_count",
    }
)

# Work output is JSON/JSONL only.  These ceilings are deliberately generous for
# a two-hour episode (normally 5-10 batches of 350-500 short records), while
# bounding all decompression before CRC testing or member reads.
MAX_TRANSLATED_ZIP_MEMBERS = 64
MAX_TRANSLATED_ZIP_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TRANSLATED_ZIP_TOTAL_BYTES = 128 * 1024 * 1024
MAX_TRANSLATED_ZIP_COMPRESSION_RATIO = 250.0
MIN_RATIO_CHECK_BYTES = 1024 * 1024


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TranslationValidationError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise TranslationValidationError(
        f"Non-finite JSON number is forbidden: {value}"
    )

DEFAULT_KNOWN_NAMES: tuple[str, ...] = (
    "Defne",
    "Kadir",
    "Tolga",
    "Levent",
    "Levent Bartıner",
    "Bartıner",
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

DEFAULT_FORBIDDEN_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "Emindağ": ("Emin Dağ", "Emin dag"),
    "Bartıner": ("Bartiner", "Bartınar"),
}

DEFAULT_SOURCE_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "Defne": ("Def", "Defneciğim", "Defter", "Medefne"),
    "Kadir": ("Kader", "Kadeh", "Katiş"),
    "Tolga": ("Tolgacım", "Tolgacığım", "Dolgu"),
    "Levent": ("Levan", "Levhat", "Elifat"),
    "Mine": (
        "Emine",
        "Mina",
        "Müniş",
        "müniş",
        "minişlere",
        "İlmi",
        "Minna",
        "Nina",
        "Emineş",
    ),
    "Melis": ("Melisa", "Menis", "Gelirse"),
    "Selim": ("Selam", "Selvi"),
    "Selma": ("Selman", "Selva"),
    "Özlem": ("özlemle",),
    "Emindağ": ("Kadir Emin", "Emin da"),
    "Bartıner": (
        "Bartner",
        "Bartilerin",
        "Bartilere",
        "partenere",
        "partnere",
        "partner",
        "Atner",
    ),
    "Suzi": ("Suzy",),
    "Oğuz": ("Oğuzhan",),
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    block_uid: str | None = None
    expected: Any = None
    actual: Any = None
    batch_file: str | None = None
    record_index: int | None = None
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "block_uid": self.block_uid,
            "expected": self.expected,
            "actual": self.actual,
            "batch_file": self.batch_file,
            "record_index": self.record_index,
            "line_number": self.line_number,
        }


@dataclass
class TranslationValidationReport:
    schema_sha256: str
    expected_block_count: int
    output_block_count: int
    tr_block_count: int = 0
    id_block_count: int = 0
    missing_translation_count: int = 0
    duplicate_translation_count: int = 0
    extra_translation_count: int = 0
    positional_translation_mismatch_count: int = 0
    schema_mismatch_count: int = 0
    uid_mismatch_count: int = 0
    order_mismatch_count: int = 0
    block_count_mismatch_count: int = 0
    timing_mismatch_count: int = 0
    episode_mismatch_count: int = 0
    schema_version_mismatch_count: int = 0
    block_index_mismatch_count: int = 0
    record_shape_mismatch_count: int = 0
    empty_text_count: int = 0
    special_name_mismatch_count: int = 0
    numeric_mismatch_count: int = 0
    religious_expression_mismatch_count: int = 0
    allah_preservation_mismatch_count: int = 0
    translation_report_mismatch_count: int = 0
    batch_mismatch_count: int = 0
    review_required_count: int = 0
    errors: list[ValidationIssue] = field(default_factory=list)

    @property
    def structural_valid(self) -> bool:
        return not any(
            (
                self.missing_translation_count,
                self.duplicate_translation_count,
                self.extra_translation_count,
                self.schema_mismatch_count,
                self.uid_mismatch_count,
                self.order_mismatch_count,
                self.block_count_mismatch_count,
                self.timing_mismatch_count,
                self.episode_mismatch_count,
                self.schema_version_mismatch_count,
                self.block_index_mismatch_count,
                self.record_shape_mismatch_count,
                self.translation_report_mismatch_count,
                self.batch_mismatch_count,
            )
        )

    @property
    def semantic_valid(self) -> bool:
        return not any(
            (
                self.positional_translation_mismatch_count,
                self.empty_text_count,
                self.special_name_mismatch_count,
                self.numeric_mismatch_count,
                self.religious_expression_mismatch_count,
                self.allah_preservation_mismatch_count,
            )
        )

    @property
    def ok(self) -> bool:
        return self.structural_valid and self.semantic_valid and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "structural_valid": self.structural_valid,
            "semantic_valid": self.semantic_valid,
            "schema_sha256": self.schema_sha256,
            "expected_block_count": self.expected_block_count,
            "output_block_count": self.output_block_count,
            "tr_block_count": self.tr_block_count,
            "id_block_count": self.id_block_count,
            "missing_translation_count": self.missing_translation_count,
            "duplicate_translation_count": self.duplicate_translation_count,
            "extra_translation_count": self.extra_translation_count,
            "positional_translation_mismatch_count": self.positional_translation_mismatch_count,
            "schema_mismatch_count": self.schema_mismatch_count,
            "uid_mismatch_count": self.uid_mismatch_count,
            "order_mismatch_count": self.order_mismatch_count,
            "block_count_mismatch_count": self.block_count_mismatch_count,
            "timing_mismatch_count": self.timing_mismatch_count,
            "episode_mismatch_count": self.episode_mismatch_count,
            "schema_version_mismatch_count": self.schema_version_mismatch_count,
            "block_index_mismatch_count": self.block_index_mismatch_count,
            "record_shape_mismatch_count": self.record_shape_mismatch_count,
            "empty_text_count": self.empty_text_count,
            "special_name_mismatch_count": self.special_name_mismatch_count,
            "numeric_mismatch_count": self.numeric_mismatch_count,
            "religious_expression_mismatch_count": self.religious_expression_mismatch_count,
            "allah_preservation_mismatch_count": self.allah_preservation_mismatch_count,
            "translation_report_mismatch_count": self.translation_report_mismatch_count,
            "batch_mismatch_count": self.batch_mismatch_count,
            "review_required_count": self.review_required_count,
            "errors": [issue.to_dict() for issue in self.errors],
        }


@dataclass
class TranslationValidationResult:
    report: TranslationValidationReport
    records_by_uid: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = {
            uid: MappingProxyType(copy.deepcopy(dict(record)))
            for uid, record in self.records_by_uid.items()
        }
        self.records_by_uid = MappingProxyType(frozen)

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def positional_translation_mismatch_count(self) -> int:
        return self.report.positional_translation_mismatch_count

    @property
    def missing_translation_count(self) -> int:
        return self.report.missing_translation_count

    @property
    def duplicate_translation_count(self) -> int:
        return self.report.duplicate_translation_count

    @property
    def extra_translation_count(self) -> int:
        return self.report.extra_translation_count

    @property
    def validated_by_uid(self) -> Mapping[str, Mapping[str, Any]]:
        """Read-only compatibility view used by FINALIZE helpers."""

        return self.records_by_uid

    def to_dict(self) -> dict[str, Any]:
        return self.report.to_dict()

    def ordered_records(
        self, schema: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Return schema order after UID lookup, never positional mapping."""

        if not self.ok:
            raise TranslationValidationError(
                "Cannot expose translations because validation failed", self
            )
        trusted = validate_episode_schema(schema)
        if trusted["schema_sha256"] != self.report.schema_sha256:
            raise TranslationValidationError(
                "Validated translations belong to a different schema", self
            )
        return [
            copy.deepcopy(dict(self.records_by_uid[block["block_uid"]]))
            for block in trusted["blocks"]
        ]


class TranslationValidationError(ValueError):
    """Hard validation failure carrying the complete structured result."""

    def __init__(
        self,
        message: str,
        result: TranslationValidationResult | None = None,
    ) -> None:
        if result is not None and result.report.errors:
            first = result.report.errors[0]
            message = f"{message}: {first.code}: {first.message}"
        super().__init__(message)
        self.result = result
        self.report = result.report if result is not None else None


@dataclass(frozen=True)
class _SourcedRecord:
    record: Mapping[str, Any]
    batch_file: str
    global_index: int
    line_number: int


def _issue(
    report: TranslationValidationReport,
    *,
    code: str,
    message: str,
    sourced: _SourcedRecord | None = None,
    block_uid: str | None = None,
    expected: Any = None,
    actual: Any = None,
    batch_file: str | None = None,
    record_index: int | None = None,
    line_number: int | None = None,
) -> None:
    report.errors.append(
        ValidationIssue(
            code=code,
            message=message,
            block_uid=(
                block_uid
                if block_uid is not None
                else (
                    sourced.record.get("block_uid")
                    if sourced is not None
                    and isinstance(sourced.record.get("block_uid"), str)
                    else None
                )
            ),
            expected=expected,
            actual=actual,
            batch_file=(
                batch_file
                if batch_file is not None
                else (sourced.batch_file if sourced is not None else None)
            ),
            record_index=(
                record_index
                if record_index is not None
                else (sourced.global_index if sourced is not None else None)
            ),
            line_number=(
                line_number
                if line_number is not None
                else (sourced.line_number if sourced is not None else None)
            ),
        )
    )


def _flatten_records(records_or_batches: Any) -> list[_SourcedRecord]:
    """Normalize supported in-memory inputs without losing duplicate records."""

    batch_entries: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    if isinstance(records_or_batches, Mapping):
        if TRANSLATION_RECORD_FIELDS.intersection(records_or_batches):
            batch_entries = [("<memory>", [records_or_batches])]
        else:
            for name, records in records_or_batches.items():
                if not isinstance(name, str) or not isinstance(records, Sequence):
                    raise TranslationValidationError(
                        "Batch mapping must map filenames to record sequences"
                    )
                batch_entries.append((name, records))
    else:
        values = list(records_or_batches)
        if not values:
            return []
        if all(isinstance(value, BatchFile) for value in values):
            batch_entries = [
                (value.name, value.records) for value in values  # type: ignore[union-attr]
            ]
        elif all(
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            for value in values
        ):
            batch_entries = [(value[0], value[1]) for value in values]
        elif all(isinstance(value, Mapping) for value in values):
            batch_entries = [("<memory>", values)]
        else:
            raise TranslationValidationError(
                "Translations must be records, BatchFile objects, or filename/records pairs"
            )

    sourced: list[_SourcedRecord] = []
    global_index = 0
    for batch_file, records in batch_entries:
        for line_number, record in enumerate(records, start=1):
            global_index += 1
            if not isinstance(record, Mapping):
                # Keep a placeholder so count/order diagnostics remain exact.
                record = {"__invalid_record__": repr(record)}
            sourced.append(
                _SourcedRecord(
                    record=record,
                    batch_file=batch_file,
                    global_index=global_index,
                    line_number=line_number,
                )
            )
    return sourced


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Whitespace is deliberately *not* an in-number separator.  Treating it as one
# made ``1 2`` indistinguishable from ``12``.  Decimal-looking punctuation is
# parsed before thousands formatting so ``1,20`` cannot collapse to ``120``.
_NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?(?!\w)",
    re.UNICODE,
)
_TURKISH_FUNCTION_WORDS = {
    "acaba",
    "ama",
    "ben",
    "bence",
    "bir",
    "biz",
    "bu",
    "cok",
    "da",
    "daha",
    "de",
    "diye",
    "icin",
    "ile",
    "mi",
    "mu",
    "ne",
    "neden",
    "nasil",
    "o",
    "sen",
    "simdi",
    "siz",
    "su",
    "ve",
}

# Canonical names may carry ordinary Turkish suffixes without an apostrophe in
# ASR text (``Defneyle``, ``Özlemciğim``, ``Emindağlar``).  Only complete,
# explicitly enumerated suffixes are accepted; arbitrary prefix matching would
# turn ordinary words into names.
_TURKISH_NAME_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            "a",
            "e",
            "i",
            "ı",
            "u",
            "ü",
            "da",
            "de",
            "dan",
            "den",
            "la",
            "le",
            "ya",
            "ye",
            "yı",
            "yi",
            "yu",
            "yü",
            "yla",
            "yle",
            "in",
            "ın",
            "un",
            "ün",
            "nin",
            "nın",
            "nun",
            "nün",
            "lar",
            "ler",
            "cı",
            "ci",
            "cu",
            "cü",
            "çı",
            "çi",
            "çu",
            "çü",
            "cım",
            "cim",
            "cum",
            "cüm",
            "çım",
            "çim",
            "çum",
            "çüm",
            "cığım",
            "ciğim",
            "cuğum",
            "cüğüm",
            "çığım",
            "çiğim",
            "çuğum",
            "çüğüm",
        },
        key=lambda value: (-len(value), value),
    )
)
_TURKISH_NAME_SUFFIX_PATTERN = "|".join(
    re.escape(value) for value in _TURKISH_NAME_SUFFIXES
)


def _semantic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = text.replace("’", "'").replace("`", "'")
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens)


def _ascii_fold(text: str) -> str:
    folded = text.casefold().translate(
        str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"})
    )
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(_TOKEN_RE.findall(folded))


def _tokens(text: str) -> list[str]:
    return _semantic_normalize(text).split()


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _ascii_fold(text).split()
        if token not in _TURKISH_FUNCTION_WORDS and not token.isdigit()
    }


def _text_similarity(left: str, right: str) -> float:
    a = _semantic_normalize(left)
    b = _semantic_normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_tokens = a.split()
    b_tokens = b.split()
    char_ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    token_order_ratio = SequenceMatcher(
        None, a_tokens, b_tokens, autojunk=False
    ).ratio()
    a_counter = Counter(a_tokens)
    b_counter = Counter(b_tokens)
    intersection = sum((a_counter & b_counter).values())
    token_dice = 2.0 * intersection / (len(a_tokens) + len(b_tokens))
    return max(
        0.65 * char_ratio + 0.35 * token_dice,
        0.55 * token_order_ratio + 0.45 * char_ratio,
    )


def _source_candidates(block: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for field_name in (
        "timing_text",
        "primary_text",
        "verification_text",
        "youtube_text",
    ):
        value = block.get(field_name)
        if isinstance(value, str) and value.strip():
            normalized = _semantic_normalize(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(value)
    return candidates


def _best_similarity(text: str, block: Mapping[str, Any]) -> float:
    return max(
        (_text_similarity(text, candidate) for candidate in _source_candidates(block)),
        default=0.0,
    )


def _extract_numbers(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        sign = ""
        if raw[:1] in {"+", "-"}:
            sign, raw = raw[0], raw[1:]
        separators = [
            position for position, character in enumerate(raw) if character in ".,"
        ]
        if not separators:
            integer, fraction = raw, ""
        else:
            last = separators[-1]
            digits_after = len(raw) - last - 1
            separator_count = len(separators)
            # One/two trailing digits are decimal evidence.  Three-digit groups
            # are formatting, so 1.250 and 1250 remain the same quantity.
            decimal = digits_after in {1, 2} and (
                separator_count == 1 or raw[last] != raw[separators[-2]]
            )
            if decimal:
                integer = re.sub(r"[.,]", "", raw[:last])
                fraction = re.sub(r"[.,]", "", raw[last + 1 :]).rstrip("0")
            else:
                integer = re.sub(r"[.,]", "", raw)
                fraction = ""
        integer = integer.lstrip("0") or "0"
        canonical = sign + integer
        if fraction:
            canonical += "." + fraction
        values.append(canonical)
    return tuple(values)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _semantic_normalize(text)
    normalized_phrase = _semantic_normalize(phrase)
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text
        )
    )


def _name_form_pattern(phrase: str, *, capture_suffix: bool = False) -> str:
    normalized_phrase = _semantic_normalize(phrase)
    suffix = (
        rf"(?P<suffix>{_TURKISH_NAME_SUFFIX_PATTERN})"
        if capture_suffix
        else rf"(?:{_TURKISH_NAME_SUFFIX_PATTERN})"
    )
    return rf"(?<!\w){re.escape(normalized_phrase)}(?:{suffix})?(?!\w)"


def _contains_name_form(text: str, name: str) -> bool:
    normalized_name = _semantic_normalize(name)
    if not normalized_name:
        return False
    return bool(re.search(_name_form_pattern(name), _semantic_normalize(text)))


def _names_in_text(text: str, names: Sequence[str]) -> frozenset[str]:
    return frozenset(name for name in names if _contains_name_form(text, name))


def _replace_name_variants(
    text: str,
    variants_by_name: Mapping[str, Sequence[str]],
) -> str:
    """Replace configured whole-phrase variants with canonical spellings."""

    normalized = _semantic_normalize(text)
    for canonical, variants in variants_by_name.items():
        canonical_text = _semantic_normalize(canonical)
        for variant in variants:
            variant_text = _semantic_normalize(variant)
            if not variant_text:
                continue
            normalized = re.sub(
                _name_form_pattern(variant_text, capture_suffix=True),
                lambda match: canonical_text + (match.group("suffix") or ""),
                normalized,
            )
    return normalized


def _source_name_sets_in_text(
    text: str,
    names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> frozenset[frozenset[str]]:
    """Return candidate-local canonical name interpretations.

    Forbidden spellings are unambiguous corrections and are canonicalized in
    every interpretation.  Source-only ASR variants are non-binding evidence:
    each may support a canonical correction without forcing it.  This matters
    for ambiguous ordinary words such as ``selam`` and ``gelirse``.
    """

    possible_texts = _source_name_text_interpretations(
        text,
        forbidden_name_variants,
        source_name_variants,
    )
    required_text = _replace_name_variants(text, forbidden_name_variants)
    required_names = _names_in_text(required_text, names)
    supported_sets: set[frozenset[str]] = set()
    for candidate in possible_texts:
        interpreted_names = _names_in_text(candidate, names)
        supported_sets.add(interpreted_names)
        # A multi-word ASR variant may overlap a genuine canonical name, as in
        # "Kadir Emin" -> Emindağ.  Preserve the literal name interpretation
        # while also allowing the correction supplied by contextual evidence.
        supported_sets.add(required_names | interpreted_names)
    return frozenset(supported_sets)


def _source_name_text_interpretations(
    text: str,
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> frozenset[str]:
    """Return mandatory and optional canonicalized source text readings."""

    required_text = _replace_name_variants(text, forbidden_name_variants)
    possible_texts = {required_text}
    for canonical, variants in source_name_variants.items():
        replacement = {canonical: variants}
        corrected_texts = {
            _replace_name_variants(candidate, replacement)
            for candidate in possible_texts
        }
        possible_texts.update(corrected_texts)
    return frozenset(possible_texts)


def _best_source_name_similarity(
    text: str,
    block: Mapping[str, Any],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> float:
    candidates = (
        interpretation
        for candidate in _source_candidates(block)
        for interpretation in _source_name_text_interpretations(
            candidate,
            forbidden_name_variants,
            source_name_variants,
        )
    )
    return max(
        (_text_similarity(text, candidate) for candidate in candidates),
        default=0.0,
    )


def _religious_anchor_set(text: str) -> frozenset[str]:
    folded = _ascii_fold(text)
    anchors: set[str] = set()
    for key, phrases in {
        "allah": ("allah", "allahim", "allah im", "vallahi"),
        "rabb": ("rabb", "rabbim", "ya rabbim", "yarabbim"),
        "insallah": ("insallah", "insyaallah"),
        "masallah": ("masallah", "masyaallah"),
        "estagfurullah": ("estagfurullah", "astagfirullah"),
        "tovbe": ("tovbe", "tobat"),
        "la_havle": ("la havle", "la hawla"),
    }.items():
        if any(
            re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded)
            for phrase in phrases
        ):
            anchors.add(key)
    return frozenset(anchors)


def _religious_evidence_set(text: str) -> frozenset[str]:
    """Return phrase-level Turkish religious evidence for candidate selection."""

    folded = _ascii_fold(text)
    anchors = set(_religious_anchor_set(text))
    # ``Vallahi`` and Indonesian ``Demi Allah`` share a language-independent
    # oath anchor for TR/ID alignment.  It is not, however, literal Turkish
    # ``Allah`` evidence for the source-to-corrected-TR phrase check.
    if re.search(r"(?<!\w)vallahi(?!\w)", folded):
        anchors.discard("allah")
    phrases = {
        "allah_askina": ("allah askina",),
        "allah_allah": ("allah allah",),
        "allahim": ("allahim", "allah im"),
        "ya_rabbim": ("ya rabbim", "yarabbim", "ya rabb im", "rabb im"),
        "tovbe_estagfurullah": ("tovbe estagfurullah",),
        "la_havle_full": ("la havle vela kuvvete illa billah",),
    }
    for key, alternatives in phrases.items():
        if any(
            re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded)
            for phrase in alternatives
        ):
            anchors.add(key)
    return frozenset(anchors)


def _alignment_anchor_signature(
    text: str, known_names: Sequence[str]
) -> tuple[frozenset[str], tuple[str, ...], frozenset[str]]:
    return (
        _names_in_text(text, known_names),
        _extract_numbers(text),
        _religious_anchor_set(text),
    )


def _anchor_similarity(
    translation: str,
    block: Mapping[str, Any],
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]] | None = None,
    source_name_variants: Mapping[str, Sequence[str]] | None = None,
) -> tuple[int, int]:
    trans_names, trans_numbers, trans_religion = _alignment_anchor_signature(
        translation, known_names
    )
    best: tuple[int, int] | None = None
    for candidate in _source_candidates(block):
        source_name_sets = _source_name_sets_in_text(
            candidate,
            known_names,
            forbidden_name_variants or {},
            source_name_variants or {},
        )
        src_numbers = _extract_numbers(candidate)
        src_religion = _religious_anchor_set(candidate)
        for src_names in source_name_sets:
            total = int(bool(trans_names or src_names)) + int(
                bool(trans_numbers or src_numbers)
            ) + int(bool(trans_religion or src_religion))
            matches = int(
                trans_names == src_names and bool(trans_names or src_names)
            )
            matches += int(
                trans_numbers == src_numbers and bool(trans_numbers or src_numbers)
            )
            matches += int(
                trans_religion == src_religion
                and bool(trans_religion or src_religion)
            )
            if best is None or matches > best[0] or (
                matches == best[0] and total < best[1]
            ):
                best = (matches, total)
    return best or (0, 0)


def _alignment_score(
    translation: str,
    block: Mapping[str, Any],
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]] | None = None,
    source_name_variants: Mapping[str, Sequence[str]] | None = None,
) -> float:
    base = max(
        _best_similarity(translation, block),
        _best_source_name_similarity(
            translation,
            block,
            forbidden_name_variants or {},
            source_name_variants or {},
        ),
    )
    matches, total = _anchor_similarity(
        translation,
        block,
        known_names,
        forbidden_name_variants,
        source_name_variants,
    )
    if total and matches == total:
        # Anchors strengthen lexical evidence but never manufacture a high
        # match score on their own.  A shared name alone is common across an
        # episode and cannot prove that dialogue belongs to another UID.
        return min(0.98, base + 0.06 * matches)
    if total and not matches:
        return max(0.0, base - 0.12)
    return base


def _turkish_evidence_supports(
    text: str,
    block: Mapping[str, Any],
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> bool:
    """Whether a corrected Turkish line remains supported by this block.

    Exact candidate text is conclusive.  Otherwise a correction needs both
    reasonable lexical continuity and one complete candidate-local anchor
    signature.  This prevents an unrelated exact line elsewhere in the episode
    from overruling evidence under the current UID.
    """

    normalized = _semantic_normalize(text)
    candidates = _source_candidates(block)
    if any(_semantic_normalize(candidate) == normalized for candidate in candidates):
        return True
    if max(
        _best_similarity(text, block),
        _best_source_name_similarity(
            text,
            block,
            forbidden_name_variants,
            source_name_variants,
        ),
    ) < 0.56:
        return False
    translated_signature = _alignment_anchor_signature(text, known_names)
    for candidate in candidates:
        for source_names in _source_name_sets_in_text(
            candidate,
            known_names,
            forbidden_name_variants,
            source_name_variants,
        ):
            source_signature = (
                source_names,
                _extract_numbers(candidate),
                _religious_anchor_set(candidate),
            )
            if translated_signature == source_signature:
                return True
    return False


def _exact_foreign_matches(
    text: str,
    normalized_sources: Mapping[str, set[int]],
) -> set[int]:
    return set(normalized_sources.get(_semantic_normalize(text), set()))


def _review_is_justified(record: Mapping[str, Any]) -> bool:
    note = record.get("note")
    return bool(
        record.get("review_required") is True
        and isinstance(note, str)
        and note.strip()
    )


def _detect_positional_mismatches(
    schema: Mapping[str, Any],
    records_by_uid: Mapping[str, Mapping[str, Any]],
    source_by_uid: Mapping[str, _SourcedRecord],
    report: TranslationValidationReport,
    *,
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
    semantic_window: int,
) -> None:
    blocks = schema["blocks"]
    normalized_sources: dict[str, set[int]] = defaultdict(set)
    for index, block in enumerate(blocks):
        for candidate in _source_candidates(block):
            normalized_sources[_semantic_normalize(candidate)].add(index)

    candidates: dict[int, tuple[int, float, float, str]] = {}
    for index, block in enumerate(blocks):
        uid = block["block_uid"]
        record = records_by_uid.get(uid)
        if record is None:
            continue
        tr_final = record.get("tr_final")
        if not isinstance(tr_final, str) or not tr_final.strip():
            continue
        current_score = _alignment_score(
            tr_final,
            block,
            known_names,
            forbidden_name_variants,
            source_name_variants,
        )
        current_supported = _turkish_evidence_supports(
            tr_final,
            block,
            known_names,
            forbidden_name_variants,
            source_name_variants,
        )
        best_index = index
        best_score = current_score
        start = max(0, index - semantic_window)
        stop = min(len(blocks), index + semantic_window + 1)
        for candidate_index in range(start, stop):
            if candidate_index == index:
                continue
            score = _alignment_score(
                tr_final,
                blocks[candidate_index],
                known_names,
                forbidden_name_variants,
                source_name_variants,
            )
            if score > best_score + 1e-12:
                best_index, best_score = candidate_index, score

        foreign_exact = _exact_foreign_matches(tr_final, normalized_sources)
        same_exact = index in foreign_exact
        all_foreign = {value for value in foreign_exact if value != index}
        unique_nearby_foreign = {
            value for value in all_foreign if abs(value - index) <= semantic_window
        }
        reason = ""
        if (
            not current_supported
            and not same_exact
            and len(unique_nearby_foreign) == 1
        ):
            exact_index = next(iter(unique_nearby_foreign))
            best_index = exact_index
            best_score = 1.0
            reason = "exact_match_to_neighbor"
        elif (
            not current_supported
            and not same_exact
            and len(all_foreign) == 1
        ):
            exact_index = next(iter(all_foreign))
            best_index = exact_index
            best_score = 1.0
            # A repeated short phrase hundreds of blocks away is not position
            # evidence.  It becomes actionable only if adjacent records prefer
            # the same displacement, like a real shifted run.
            reason = "sequence_candidate"
        elif (
            best_index != index
            and not current_supported
            and best_score >= 0.78
            and best_score - current_score >= 0.18
        ):
            reason = "neighbor_similarity"
        else:
            current_anchors = _alignment_anchor_signature(tr_final, known_names)
            has_anchor = any(current_anchors)
            if has_anchor and best_index != index and not current_supported:
                best_matches, best_total = _anchor_similarity(
                    tr_final,
                    blocks[best_index],
                    known_names,
                    forbidden_name_variants,
                    source_name_variants,
                )
                current_matches, current_total = _anchor_similarity(
                    tr_final,
                    block,
                    known_names,
                    forbidden_name_variants,
                    source_name_variants,
                )
                if (
                    best_total
                    and best_matches == best_total
                    and best_matches > current_matches
                    and best_score >= 0.55
                ):
                    reason = "name_number_or_religious_anchor"

        if not reason:
            evidence_content: set[str] = set()
            for source_candidate in _source_candidates(block):
                evidence_content.update(_content_tokens(source_candidate))
            translated_content = _content_tokens(tr_final)
            if (
                len(evidence_content) >= 2
                and len(translated_content) >= 2
                and evidence_content.isdisjoint(translated_content)
                and current_score < 0.36
            ):
                # Turkish correction should remain evidential, not paraphrase or
                # invent unrelated dialogue.  With no shared content token and
                # very low string similarity, the current UID is unsupported
                # even when the originating block is outside the local window.
                best_index = index
                best_score = current_score
                reason = "unsupported_turkish_dialogue"

        if reason:
            candidates[index] = (best_index, current_score, best_score, reason)
        elif (
            best_index != index
            and best_score >= 0.62
            and best_score - current_score >= 0.10
        ):
            # A moderate candidate becomes actionable only as part of a run.
            candidates[index] = (
                best_index,
                current_score,
                best_score,
                "sequence_candidate",
            )

    # Mark strong candidates immediately.  Moderate candidates require at least
    # two consecutive blocks preferring the same displacement, which is the
    # signature of the historical shift rather than an isolated correction.
    flagged: dict[int, tuple[int, float, float, str]] = {
        index: value
        for index, value in candidates.items()
        if value[3] != "sequence_candidate"
        and not (
            value[3]
            in {
                "unsupported_turkish_dialogue",
                "neighbor_similarity",
                "name_number_or_religious_anchor",
            }
            and _review_is_justified(
                records_by_uid[blocks[index]["block_uid"]]
            )
        )
    }
    sorted_indices = sorted(candidates)
    run: list[int] = []
    run_offset: int | None = None
    for index in sorted_indices + [len(blocks) + semantic_window + 1]:
        value = candidates.get(index)
        offset = value[0] - index if value is not None else None
        if (
            run
            and value is not None
            and index == run[-1] + 1
            and offset == run_offset
        ):
            run.append(index)
            continue
        if len(run) >= 2:
            for run_index in run:
                flagged[run_index] = candidates[run_index]
        run = [index] if value is not None else []
        run_offset = offset
    if len(run) >= 2:  # Defensive; sentinel normally flushes it.
        for run_index in run:
            flagged[run_index] = candidates[run_index]

    report.positional_translation_mismatch_count = len(flagged)
    for index, (matched_index, current_score, matched_score, reason) in sorted(
        flagged.items()
    ):
        block = blocks[index]
        matched_block = blocks[matched_index]
        uid = block["block_uid"]
        sourced = source_by_uid.get(uid)
        actual_text = records_by_uid[uid].get("tr_final")
        if matched_index == index and reason == "unsupported_turkish_dialogue":
            message = (
                f"tr_final under block {block['block_index']} has insufficient "
                "support in that block's Turkish evidence"
            )
        else:
            message = (
                f"tr_final under block {block['block_index']} matches block "
                f"{matched_block['block_index']} more strongly ({reason})"
            )
        _issue(
            report,
            code="positional_translation_mismatch",
            message=message,
            sourced=sourced,
            block_uid=uid,
            expected={
                "block_index": block["block_index"],
                "source_evidence": _source_candidates(block),
                "same_block_score": round(current_score, 4),
            },
            actual={
                "tr_final": actual_text,
                "likely_source_block_uid": matched_block["block_uid"],
                "likely_source_block_index": matched_block["block_index"],
                "neighbor_score": round(matched_score, 4),
            },
        )


def _source_anchor_signatures(
    block: Mapping[str, Any],
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> set[tuple[frozenset[str], tuple[str, ...], frozenset[str]]]:
    return {
        (
            source_names,
            _extract_numbers(candidate),
            _religious_anchor_set(candidate),
        )
        for candidate in _source_candidates(block)
        for source_names in _source_name_sets_in_text(
            candidate,
            known_names,
            forbidden_name_variants,
            source_name_variants,
        )
    }


def _detect_indonesian_anchor_mismatches(
    schema: Mapping[str, Any],
    records_by_uid: Mapping[str, Mapping[str, Any]],
    source_by_uid: Mapping[str, _SourcedRecord],
    report: TranslationValidationReport,
    *,
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
    semantic_window: int,
) -> None:
    """Check ``id_final`` position using only language-independent anchors.

    Names, exact numeric tokens and canonical religious anchors can prove that
    an Indonesian line belongs to another block.  Anchor-free Indonesian prose
    cannot be mapped to Turkish evidence without a translation model/API, so it
    is deliberately not guessed at here.
    """

    blocks = schema["blocks"]
    translated_signatures: list[
        tuple[frozenset[str], tuple[str, ...], frozenset[str]] | None
    ] = []
    for block in blocks:
        record = records_by_uid.get(block["block_uid"])
        tr_final = record.get("tr_final") if record is not None else None
        translated_signatures.append(
            _alignment_anchor_signature(tr_final, known_names)
            if isinstance(tr_final, str) and tr_final.strip()
            else None
        )
    flagged: list[tuple[int, int, tuple[frozenset[str], tuple[str, ...], frozenset[str]]]] = []
    for index, block in enumerate(blocks):
        uid = block["block_uid"]
        record = records_by_uid.get(uid)
        if record is None:
            continue
        id_final = record.get("id_final")
        if not isinstance(id_final, str) or not id_final.strip():
            continue
        id_signature = _alignment_anchor_signature(id_final, known_names)
        # The Indonesian line translates the corrected Turkish selected under
        # this UID, not the noisy raw ASR.  Matching TR/ID anchors are therefore
        # conclusive same-block evidence.
        if not any(id_signature) or id_signature == translated_signatures[index]:
            continue
        if sum(len(category) for category in id_signature) < 2:
            # One recurring name or oath is not a positional identity.  Strict
            # TR/ID parity below still rejects an anchor added or dropped under
            # the current UID.
            continue

        foreign_matches = {
            candidate_index
            for candidate_index, candidate_signature in enumerate(
                translated_signatures
            )
            if candidate_index != index and id_signature == candidate_signature
        }
        nearby_matches = {
            candidate_index
            for candidate_index in foreign_matches
            if abs(candidate_index - index) <= semantic_window
        }
        if len(nearby_matches) == 1:
            matched_index = next(iter(nearby_matches))
        elif len(foreign_matches) == 1:
            matched_index = next(iter(foreign_matches))
        else:
            # Repeated anchors are not a safe positional identity signal.
            continue
        flagged.append((index, matched_index, id_signature))

    report.positional_translation_mismatch_count += len(flagged)
    for index, matched_index, signature in flagged:
        block = blocks[index]
        matched_block = blocks[matched_index]
        uid = block["block_uid"]
        sourced = source_by_uid.get(uid)
        _issue(
            report,
            code="positional_translation_mismatch",
            message=(
                f"id_final under block {block['block_index']} has exact preserved "
                f"anchors from block {matched_block['block_index']}"
            ),
            sourced=sourced,
            block_uid=uid,
            expected={
                "field": "id_final",
                "block_index": block["block_index"],
                "supported_anchor_signatures": (
                    []
                    if translated_signatures[index] is None
                    else [
                        {
                            "names": sorted(translated_signatures[index][0]),
                            "numbers": list(translated_signatures[index][1]),
                            "religious": sorted(translated_signatures[index][2]),
                        }
                    ]
                ),
            },
            actual={
                "field": "id_final",
                "id_final": records_by_uid[uid].get("id_final"),
                "names": sorted(signature[0]),
                "numbers": list(signature[1]),
                "religious": sorted(signature[2]),
                "likely_source_block_uid": matched_block["block_uid"],
                "likely_source_block_index": matched_block["block_index"],
                "reason": "exact_preserved_anchor_match",
            },
        )


def _candidate_number_sets(block: Mapping[str, Any]) -> set[tuple[str, ...]]:
    return {_extract_numbers(candidate) for candidate in _source_candidates(block)}


def _validate_content_anchors(
    schema: Mapping[str, Any],
    records_by_uid: Mapping[str, Mapping[str, Any]],
    source_by_uid: Mapping[str, _SourcedRecord],
    report: TranslationValidationReport,
    *,
    known_names: Sequence[str],
    forbidden_name_variants: Mapping[str, Sequence[str]],
    source_name_variants: Mapping[str, Sequence[str]],
) -> None:
    for block in schema["blocks"]:
        uid = block["block_uid"]
        record = records_by_uid.get(uid)
        if record is None:
            continue
        sourced = source_by_uid.get(uid)
        tr_final = record.get("tr_final")
        id_final = record.get("id_final")
        if not isinstance(tr_final, str) or not isinstance(id_final, str):
            continue

        # Number/money preservation: corrected Turkish may select any complete
        # candidate-local number sequence, including the empty sequence.  The
        # latter is important: it rejects a quantity invented in both outputs.
        evidence_number_sets = _candidate_number_sets(block)
        tr_numbers = _extract_numbers(tr_final)
        id_numbers = _extract_numbers(id_final)
        number_error = False
        if tr_numbers not in evidence_number_sets:
            number_error = True
        if id_numbers != tr_numbers:
            number_error = True
        if number_error:
            report.numeric_mismatch_count += 1
            _issue(
                report,
                code="numeric_mismatch",
                message="Numbers or money values changed within this block",
                sourced=sourced,
                block_uid=uid,
                expected={
                    "supported_source_numbers": [
                        list(values) for values in sorted(evidence_number_sets)
                    ],
                    "tr_final_numbers": list(tr_numbers),
                },
                actual={"id_final_numbers": list(id_numbers)},
            )

        source_candidates = _source_candidates(block)
        supported_name_sets = {
            source_names
            for candidate in source_candidates
            for source_names in _source_name_sets_in_text(
                candidate,
                known_names,
                forbidden_name_variants,
                source_name_variants,
            )
        }
        tr_names = _names_in_text(tr_final, known_names)
        id_names = _names_in_text(id_final, known_names)
        bad_variants: list[dict[str, str]] = []
        for canonical, variants in forbidden_name_variants.items():
            for variant in variants:
                if _contains_name_form(
                    tr_final, variant
                ) or _contains_name_form(
                    id_final, variant
                ):
                    bad_variants.append(
                        {"canonical": canonical, "forbidden_variant": variant}
                    )
        source_supports_tr = tr_names in supported_name_sets
        tr_id_names_match = id_names == tr_names
        name_error = bool(
            bad_variants
            or not tr_id_names_match
            or (
                not source_supports_tr
                and not _review_is_justified(record)
            )
        )
        if name_error:
            report.special_name_mismatch_count += 1
            _issue(
                report,
                code="special_name_mismatch",
                message="A protected name is missing, translated or misspelled",
                sourced=sourced,
                block_uid=uid,
                expected={
                    "supported_source_name_sets": [
                        sorted(name_set)
                        for name_set in sorted(
                            supported_name_sets,
                            key=lambda value: sorted(value),
                        )
                    ],
                    "tr_final_names": sorted(tr_names),
                    "review_can_defer_source_ambiguity": _review_is_justified(
                        record
                    ),
                },
                actual={
                    "id_final_names": sorted(id_names),
                    "tr_id_names_match": tr_id_names_match,
                    "source_supports_tr": source_supports_tr,
                    "forbidden_variants": bad_variants,
                },
            )

        supported_religious_sets = {
            _religious_evidence_set(candidate)
            for candidate in source_candidates
        }
        tr_religious = _religious_evidence_set(tr_final)
        if (
            tr_religious not in supported_religious_sets
            and not _review_is_justified(record)
        ):
            report.religious_expression_mismatch_count += 1
            _issue(
                report,
                code="religious_source_evidence_mismatch",
                message=(
                    "Corrected Turkish religious expressions do not match any "
                    "single source candidate"
                ),
                sourced=sourced,
                block_uid=uid,
                expected=[
                    sorted(values)
                    for values in sorted(
                        supported_religious_sets,
                        key=lambda value: sorted(value),
                    )
                ],
                actual=sorted(tr_religious),
            )

        _validate_religious_expressions(
            # tr_final is the selected, corrected Turkish hypothesis.  Feeding
            # the union of every ASR candidate here would impose contradictory
            # phrase mappings that the translator could not simultaneously meet.
            source_text="",
            tr_final=tr_final,
            id_final=id_final,
            uid=uid,
            sourced=sourced,
            report=report,
        )


def _has_words(text: str, words: Sequence[str]) -> bool:
    normalized = _ascii_fold(text)
    return all(
        re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized)
        for word in words
    )


def _validate_religious_expressions(
    *,
    source_text: str,
    tr_final: str,
    id_final: str,
    uid: str,
    sourced: _SourcedRecord | None,
    report: TranslationValidationReport,
) -> None:
    turkish = _ascii_fold(f"{source_text} {tr_final}")
    indonesian = _ascii_fold(id_final)
    failures: list[dict[str, Any]] = []

    checks: list[tuple[str, bool, Sequence[Sequence[str]]]] = [
        (
            "Allah aşkına",
            "allah askina" in turkish,
            (("demi", "allah"),),
        ),
        (
            "Allah Allah / Allah'ım",
            "allah allah" in turkish or "allah im" in turkish or "allahim" in turkish,
            (("ya", "allah"),),
        ),
        (
            "Ya Rabbim",
            (
                "ya rabbim" in turkish
                or "yarabbim" in turkish
                or "ya rabb im" in turkish
                or "rabb im" in turkish
            ),
            (("ya", "rabb"),),
        ),
        ("İnşallah", "insallah" in turkish, (("insyaallah",),)),
        ("Maşallah", "masallah" in turkish, (("masyaallah",),)),
        (
            "Tövbe estağfurullah",
            "tovbe estagfurullah" in turkish,
            (("tobat", "astagfirullah"),),
        ),
        (
            "Estağfurullah",
            "estagfurullah" in turkish,
            (("astagfirullah",),),
        ),
        (
            "La havle vela kuvvete illa billah",
            "la havle vela kuvvete illa billah" in turkish,
            (("la", "hawla", "wala", "quwwata", "illa", "billah"),),
        ),
    ]
    for source_phrase, applies, alternatives in checks:
        if applies and not any(_has_words(indonesian, words) for words in alternatives):
            failures.append(
                {
                    "source_expression": source_phrase,
                    "required_indonesian_words": [list(words) for words in alternatives],
                }
            )
    if failures:
        report.religious_expression_mismatch_count += 1
        _issue(
            report,
            code="religious_expression_mismatch",
            message="Mandatory religious expression mapping was not preserved",
            sourced=sourced,
            block_uid=uid,
            expected=failures,
            actual=id_final,
        )

    contains_allah = bool(re.search(r"(?<!\w)allah(?:im)?(?!\w)", turkish))
    id_contains_allah = bool(re.search(r"allah", indonesian))
    if contains_allah and not id_contains_allah:
        report.allah_preservation_mismatch_count += 1
        _issue(
            report,
            code="allah_preservation_mismatch",
            message="Turkish contains Allah but Indonesian does not preserve Allah",
            sourced=sourced,
            block_uid=uid,
            expected="Indonesian containing Allah (for example 'Demi Allah' or 'Semoga Allah')",
            actual=id_final,
        )


def _validate_translation_report_document(
    document: Mapping[str, Any] | None,
    report: TranslationValidationReport,
    *,
    actual_output_count: int,
) -> None:
    if document is None:
        return
    if not isinstance(document, Mapping):
        report.translation_report_mismatch_count += 1
        _issue(
            report,
            code="translation_report_invalid",
            message="translation_report.json must contain a JSON object",
            expected=sorted(TRANSLATION_REPORT_FIELDS),
            actual=type(document).__name__,
        )
        return
    actual_fields = set(document)
    if actual_fields != TRANSLATION_REPORT_FIELDS:
        report.translation_report_mismatch_count += 1
        _issue(
            report,
            code="translation_report_shape_mismatch",
            message="translation_report.json fields do not match the contract",
            expected=sorted(TRANSLATION_REPORT_FIELDS),
            actual=sorted(actual_fields),
        )
    expected_values = {
        "schema_sha256": report.schema_sha256,
        "total_input_blocks": report.expected_block_count,
        "total_output_blocks": actual_output_count,
        "missing_block_count": report.missing_translation_count,
        "duplicate_block_count": report.duplicate_translation_count,
        "review_required_count": report.review_required_count,
    }
    for key, expected in expected_values.items():
        actual = document.get(key)
        if actual != expected:
            report.translation_report_mismatch_count += 1
            if key == "schema_sha256":
                report.schema_mismatch_count += 1
            _issue(
                report,
                code="translation_report_value_mismatch",
                message=f"translation_report.json {key} does not match actual output",
                expected=expected,
                actual=actual,
            )


def validate_translation_records(
    schema: Mapping[str, Any],
    records_or_batches: Any,
    *,
    translation_report: Mapping[str, Any] | None = None,
    known_names: Sequence[str] | None = None,
    forbidden_name_variants: Mapping[str, Sequence[str]] | None = None,
    source_name_variants: Mapping[str, Sequence[str]] | None = None,
    semantic_window: int = 6,
    expected_batch_uids: Mapping[str, Sequence[str]] | None = None,
    raise_on_error: bool = True,
) -> TranslationValidationResult:
    """Validate translation structure, identity, content anchors and alignment.

    ``source_name_variants`` are optional source interpretations; they never
    enter the final forbidden-spelling scan.  ``forbidden_name_variants`` are
    source correction evidence and remain prohibited in both final languages.

    On failure the default behavior is a hard exception.  Pass
    ``raise_on_error=False`` only for diagnostics/tests; no translations map is
    exposed unless the complete UID structure is safe.
    """

    try:
        trusted = validate_episode_schema(schema)
    except SchemaError as exc:
        raise TranslationValidationError(f"Invalid trusted schema: {exc}") from exc
    if isinstance(semantic_window, bool) or not isinstance(semantic_window, int) or semantic_window < 1:
        raise TranslationValidationError("semantic_window must be a positive integer")
    names = tuple(DEFAULT_KNOWN_NAMES if known_names is None else known_names)
    forbidden = (
        DEFAULT_FORBIDDEN_NAME_VARIANTS
        if forbidden_name_variants is None
        else forbidden_name_variants
    )
    source_variants = (
        DEFAULT_SOURCE_NAME_VARIANTS
        if source_name_variants is None
        else source_name_variants
    )
    sourced_records = _flatten_records(records_or_batches)
    report = TranslationValidationReport(
        schema_sha256=trusted["schema_sha256"],
        expected_block_count=trusted["block_count"],
        output_block_count=len(sourced_records),
    )

    expected_uids = [block["block_uid"] for block in trusted["blocks"]]
    expected_uid_set = set(expected_uids)
    block_by_uid = {block["block_uid"]: block for block in trusted["blocks"]}

    actual_uids: list[str | None] = []
    uid_occurrences: dict[str, list[_SourcedRecord]] = defaultdict(list)
    structurally_well_typed: dict[int, bool] = {}
    for sourced in sourced_records:
        record = sourced.record
        fields = set(record)
        if fields != TRANSLATION_RECORD_FIELDS:
            report.record_shape_mismatch_count += 1
            _issue(
                report,
                code="translation_record_shape_mismatch",
                message="Translation record fields do not match the exact contract",
                sourced=sourced,
                expected=sorted(TRANSLATION_RECORD_FIELDS),
                actual=sorted(fields),
            )

        uid = record.get("block_uid")
        uid_valid = isinstance(uid, str) and bool(uid.strip())
        if not uid_valid:
            report.uid_mismatch_count += 1
            actual_uids.append(None)
            _issue(
                report,
                code="missing_or_invalid_block_uid",
                message="block_uid must be a non-empty string",
                sourced=sourced,
                expected="non-empty schema block_uid",
                actual=uid,
            )
        else:
            actual_uids.append(uid)
            uid_occurrences[uid].append(sourced)

        supplied_sha = record.get("schema_sha256")
        if supplied_sha != trusted["schema_sha256"]:
            report.schema_mismatch_count += 1
            _issue(
                report,
                code="schema_sha256_mismatch",
                message="Translation record belongs to another or corrupt schema",
                sourced=sourced,
                expected=trusted["schema_sha256"],
                actual=supplied_sha,
            )

        type_errors: list[str] = []
        if not isinstance(record.get("tr_final"), str):
            type_errors.append("tr_final must be a string")
        if not isinstance(record.get("id_final"), str):
            type_errors.append("id_final must be a string")
        if not isinstance(record.get("review_required"), bool):
            type_errors.append("review_required must be a boolean")
        if not isinstance(record.get("note"), str):
            type_errors.append("note must be a string")
        if type_errors:
            report.record_shape_mismatch_count += 1
            _issue(
                report,
                code="translation_record_type_mismatch",
                message="; ".join(type_errors),
                sourced=sourced,
            )
            structurally_well_typed[sourced.global_index] = False
        else:
            structurally_well_typed[sourced.global_index] = True
            if record["review_required"]:
                report.review_required_count += 1
            empty_fields = [
                field_name
                for field_name in ("tr_final", "id_final")
                if not record[field_name].strip()
            ]
            if empty_fields:
                report.empty_text_count += 1
                _issue(
                    report,
                    code="empty_translation",
                    message="Required translation text is empty",
                    sourced=sourced,
                    expected="non-empty tr_final and id_final",
                    actual=empty_fields,
                )

        # Timing/index/episode/version are forbidden by the exact output shape.
        # If supplied anyway, report their attempted mutation precisely.
        if uid_valid and uid in block_by_uid:
            expected_block = block_by_uid[uid]
            for field_name in ("start_ms", "end_ms"):
                if field_name in record and record.get(field_name) != expected_block[field_name]:
                    report.timing_mismatch_count += 1
                    _issue(
                        report,
                        code="timing_mismatch",
                        message=f"Translator attempted to change {field_name}",
                        sourced=sourced,
                        expected=expected_block[field_name],
                        actual=record.get(field_name),
                    )
            if "block_index" in record and record.get("block_index") != expected_block["block_index"]:
                report.block_index_mismatch_count += 1
                _issue(
                    report,
                    code="block_index_mismatch",
                    message="Translator attempted to renumber a block",
                    sourced=sourced,
                    expected=expected_block["block_index"],
                    actual=record.get("block_index"),
                )
            if "episode" in record and record.get("episode") != trusted["episode"]:
                report.episode_mismatch_count += 1
                _issue(
                    report,
                    code="episode_mismatch",
                    message="Translation record declares the wrong episode",
                    sourced=sourced,
                    expected=trusted["episode"],
                    actual=record.get("episode"),
                )
            if "schema_version" in record and record.get("schema_version") != trusted["schema_version"]:
                report.schema_version_mismatch_count += 1
                _issue(
                    report,
                    code="schema_version_mismatch",
                    message="Translation record declares the wrong schema version",
                    sourced=sourced,
                    expected=trusted["schema_version"],
                    actual=record.get("schema_version"),
                )
        elif uid_valid:
            episode_match = re.match(r"^MA(\d+)-", uid)
            if episode_match and int(episode_match.group(1)) != trusted["episode"]:
                report.episode_mismatch_count += 1
                _issue(
                    report,
                    code="episode_mismatch",
                    message="block_uid prefix belongs to another episode",
                    sourced=sourced,
                    expected=trusted["episode"],
                    actual=int(episode_match.group(1)),
                )

    duplicate_uids = {
        uid: occurrences
        for uid, occurrences in uid_occurrences.items()
        if len(occurrences) > 1
    }
    report.duplicate_translation_count = sum(
        len(occurrences) - 1 for occurrences in duplicate_uids.values()
    )
    for uid, occurrences in duplicate_uids.items():
        for duplicate in occurrences[1:]:
            _issue(
                report,
                code="duplicate_translation_uid",
                message="block_uid occurs more than once in translation output",
                sourced=duplicate,
                block_uid=uid,
                expected="exactly one occurrence",
                actual=len(occurrences),
            )

    actual_uid_set = set(uid_occurrences)
    missing_uids = [uid for uid in expected_uids if uid not in actual_uid_set]
    extra_uids = [uid for uid in uid_occurrences if uid not in expected_uid_set]
    expected_batch_for_uid: dict[str, str] = {}
    if expected_batch_uids is not None:
        for batch_name, batch_sequence in expected_batch_uids.items():
            for batch_uid in batch_sequence:
                expected_batch_for_uid[str(batch_uid)] = batch_name
    report.missing_translation_count = len(missing_uids)
    report.extra_translation_count = len(extra_uids)
    for uid in missing_uids:
        expected_block = block_by_uid[uid]
        _issue(
            report,
            code="missing_translation_uid",
            message="Schema block has no translation record",
            block_uid=uid,
            expected={
                "block_index": expected_block["block_index"],
                "schema_sha256": trusted["schema_sha256"],
            },
            actual=None,
            batch_file=expected_batch_for_uid.get(uid),
        )
    for uid in extra_uids:
        sourced = uid_occurrences[uid][0]
        _issue(
            report,
            code="extra_translation_uid",
            message="Translation output contains a UID absent from this schema",
            sourced=sourced,
            block_uid=uid,
            expected="UID from trusted schema",
            actual=uid,
        )

    if len(sourced_records) != trusted["block_count"]:
        report.block_count_mismatch_count = abs(
            len(sourced_records) - trusted["block_count"]
        ) or 1
        _issue(
            report,
            code="block_count_mismatch",
            message="Translation block count differs from immutable schema",
            expected=trusted["block_count"],
            actual=len(sourced_records),
        )

    # Position comparison is diagnostic only.  It never creates associations.
    max_length = max(len(expected_uids), len(actual_uids))
    order_mismatch_positions: list[int] = []
    for index in range(max_length):
        expected = expected_uids[index] if index < len(expected_uids) else None
        actual = actual_uids[index] if index < len(actual_uids) else None
        if expected != actual:
            order_mismatch_positions.append(index)
            sourced = sourced_records[index] if index < len(sourced_records) else None
            _issue(
                report,
                code="translation_order_mismatch",
                message=f"UID order mismatch at global position {index + 1}",
                sourced=sourced,
                block_uid=expected or (actual if isinstance(actual, str) else None),
                expected=expected,
                actual=actual,
            )
    report.order_mismatch_count = len(order_mismatch_positions)
    report.uid_mismatch_count += (
        len(missing_uids)
        + len(extra_uids)
        + report.duplicate_translation_count
        + report.order_mismatch_count
    )

    if expected_batch_uids is not None:
        actual_by_file: dict[str, list[str | None]] = defaultdict(list)
        for sourced, uid in zip(sourced_records, actual_uids):
            actual_by_file[sourced.batch_file].append(uid)
        expected_names = list(expected_batch_uids)
        if list(actual_by_file) != expected_names:
            report.batch_mismatch_count += 1
            _issue(
                report,
                code="translated_batch_set_or_order_mismatch",
                message="Translated batch filenames/order do not match the input pack",
                expected=expected_names,
                actual=list(actual_by_file),
            )
        for batch_name, expected_batch_sequence in expected_batch_uids.items():
            actual_batch_sequence = actual_by_file.get(batch_name, [])
            if list(expected_batch_sequence) != actual_batch_sequence:
                report.batch_mismatch_count += 1
                _issue(
                    report,
                    code="translated_batch_membership_mismatch",
                    message=f"UID membership/order mismatch in {batch_name}",
                    expected=list(expected_batch_sequence),
                    actual=actual_batch_sequence,
                )

    report.tr_block_count = sum(
        1
        for sourced in sourced_records
        if isinstance(sourced.record.get("tr_final"), str)
    )
    report.id_block_count = sum(
        1
        for sourced in sourced_records
        if isinstance(sourced.record.get("id_final"), str)
    )

    # Build a private UID map only for unique, expected records.  It is exposed
    # to FINALIZE only if the whole report passes.
    candidate_map: dict[str, dict[str, Any]] = {}
    source_by_uid: dict[str, _SourcedRecord] = {}
    for uid in expected_uids:
        occurrences = uid_occurrences.get(uid, [])
        if len(occurrences) == 1:
            sourced = occurrences[0]
            candidate_map[uid] = copy.deepcopy(dict(sourced.record))
            source_by_uid[uid] = sourced

    _detect_positional_mismatches(
        trusted,
        candidate_map,
        source_by_uid,
        report,
        known_names=names,
        forbidden_name_variants=forbidden,
        source_name_variants=source_variants,
        semantic_window=semantic_window,
    )
    _detect_indonesian_anchor_mismatches(
        trusted,
        candidate_map,
        source_by_uid,
        report,
        known_names=names,
        forbidden_name_variants=forbidden,
        source_name_variants=source_variants,
        semantic_window=semantic_window,
    )
    _validate_content_anchors(
        trusted,
        candidate_map,
        source_by_uid,
        report,
        known_names=names,
        forbidden_name_variants=forbidden,
        source_name_variants=source_variants,
    )
    _validate_translation_report_document(
        translation_report,
        report,
        actual_output_count=len(sourced_records),
    )

    safe_map = candidate_map if report.ok else {}
    result = TranslationValidationResult(report=report, records_by_uid=safe_map)
    if raise_on_error and not result.ok:
        raise TranslationValidationError("Translation validation failed", result)
    return result


def _safe_zip_members(archive: zipfile.ZipFile) -> list[str]:
    infos = archive.infolist()
    if len(infos) > MAX_TRANSLATED_ZIP_MEMBERS:
        raise TranslationValidationError(
            "Translated ZIP has too many members: "
            f"{len(infos)} > {MAX_TRANSLATED_ZIP_MEMBERS}"
        )
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TranslationValidationError(
            "Translated ZIP contains duplicate member names"
        )
    total_uncompressed = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        if (
            not name
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or len(pure.parts) != 1
            or info.is_dir()
        ):
            raise TranslationValidationError(
                f"Unsafe or nested translated ZIP member: {name}"
            )
        if info.flag_bits & 0x1:
            raise TranslationValidationError(
                f"Encrypted translated ZIP member is forbidden: {name}"
            )
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise TranslationValidationError(
                f"Unsupported compression method for translated ZIP member: {name}"
            )
        if info.file_size > MAX_TRANSLATED_ZIP_MEMBER_BYTES:
            raise TranslationValidationError(
                f"Translated ZIP member is too large: {name} "
                f"({info.file_size} > {MAX_TRANSLATED_ZIP_MEMBER_BYTES} bytes)"
            )
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_TRANSLATED_ZIP_TOTAL_BYTES:
            raise TranslationValidationError(
                "Translated ZIP total uncompressed size is too large: "
                f"{total_uncompressed} > {MAX_TRANSLATED_ZIP_TOTAL_BYTES} bytes"
            )
        if info.file_size >= MIN_RATIO_CHECK_BYTES:
            if info.compress_size <= 0:
                raise TranslationValidationError(
                    f"Translated ZIP member has an invalid compression size: {name}"
                )
            ratio = info.file_size / info.compress_size
            if ratio > MAX_TRANSLATED_ZIP_COMPRESSION_RATIO:
                raise TranslationValidationError(
                    f"Translated ZIP member has a suspicious compression ratio: "
                    f"{name} ({ratio:.1f}:1)"
                )
    return names


def _verify_zip_crc(archive: zipfile.ZipFile) -> None:
    try:
        bad_member = archive.testzip()
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
        raise TranslationValidationError(
            "Translated ZIP decompression/CRC verification failed"
        ) from exc
    if bad_member is not None:
        raise TranslationValidationError(
            f"Translated ZIP CRC failure: {bad_member}"
        )


def _decode_json(payload: bytes, member: str) -> Any:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TranslationValidationError(f"{member} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise TranslationValidationError(
            f"Invalid JSON in {member} line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _decode_jsonl(payload: bytes, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TranslationValidationError(f"{member} is not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise TranslationValidationError(
                f"Invalid JSONL in {member} line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise TranslationValidationError(
                f"{member} line {line_number} must be a JSON object"
            )
        records.append(record)
    return records


def _load_manifest_value(
    value: Mapping[str, Any] | str | os.PathLike[str] | None,
    *,
    expected_schema: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    if path.suffix.lower() == ".zip":
        try:
            return validate_translation_pack(
                path, expected_schema=expected_schema
            )
        except (BatchError, zipfile.BadZipFile, KeyError) as exc:
            raise TranslationValidationError(
                f"Cannot read input manifest from translation pack: {path}"
            ) from exc
    try:
        return _decode_json(path.read_bytes(), path.name)
    except OSError as exc:
        raise TranslationValidationError(f"Cannot read input manifest: {path}") from exc


def _expected_batches(
    schema: Mapping[str, Any],
    input_manifest: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    if input_manifest is None:
        batches = build_translation_batches(schema)
        names = [f"translated_{batch.name}" for batch in batches]
        return names, {
            f"translated_{batch.name}": [record["block_uid"] for record in batch.records]
            for batch in batches
        }

    for key in ("episode", "schema_version", "schema_sha256", "block_count"):
        if input_manifest.get(key) != schema[key]:
            raise TranslationValidationError(
                f"Input manifest {key} does not match trusted schema: "
                f"expected {schema[key]!r}, got {input_manifest.get(key)!r}"
            )
    descriptors = input_manifest.get("batches")
    if not isinstance(descriptors, list) or not descriptors:
        raise TranslationValidationError("Input manifest has no valid batches list")
    expected_names: list[str] = []
    expected_uids: dict[str, list[str]] = {}
    blocks = schema["blocks"]
    cursor = 0
    for number, descriptor in enumerate(descriptors, start=1):
        if not isinstance(descriptor, Mapping):
            raise TranslationValidationError("Input manifest batch must be an object")
        expected_input = f"batch_{number:03d}.jsonl"
        expected_output = f"translated_batch_{number:03d}.jsonl"
        if descriptor.get("input_file") != expected_input:
            raise TranslationValidationError(
                f"Input manifest batch sequence mismatch at {number:03d}"
            )
        output_name = descriptor.get("output_file", expected_output)
        if output_name != expected_output:
            raise TranslationValidationError(
                f"Input manifest output filename mismatch at {number:03d}"
            )
        count = descriptor.get("block_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise TranslationValidationError(
                f"Invalid block_count for input batch {number:03d}"
            )
        sequence = [
            block["block_uid"] for block in blocks[cursor : cursor + count]
        ]
        if len(sequence) != count:
            raise TranslationValidationError("Input manifest batch counts exceed schema")
        if descriptor.get("first_block_uid") not in (None, sequence[0]):
            raise TranslationValidationError(
                f"Input manifest first UID mismatch in {expected_input}"
            )
        if descriptor.get("last_block_uid") not in (None, sequence[-1]):
            raise TranslationValidationError(
                f"Input manifest last UID mismatch in {expected_input}"
            )
        expected_names.append(expected_output)
        expected_uids[expected_output] = sequence
        cursor += count
    if cursor != schema["block_count"]:
        raise TranslationValidationError(
            "Input manifest batch counts do not cover the complete schema"
        )
    return expected_names, expected_uids


def load_and_validate_translated_zip(
    schema: Mapping[str, Any],
    translated_zip: str | os.PathLike[str],
    *,
    input_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    known_names: Sequence[str] | None = None,
    forbidden_name_variants: Mapping[str, Sequence[str]] | None = None,
    source_name_variants: Mapping[str, Sequence[str]] | None = None,
    semantic_window: int = 6,
    raise_on_error: bool = True,
) -> TranslationValidationResult:
    """Load a Work ZIP, verify its exact topology/CRC, then validate all records."""

    try:
        trusted = validate_episode_schema(schema)
    except SchemaError as exc:
        raise TranslationValidationError(f"Invalid trusted schema: {exc}") from exc
    manifest = _load_manifest_value(input_manifest, expected_schema=trusted)
    expected_names, expected_batch_uids = _expected_batches(trusted, manifest)
    path = Path(translated_zip)
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise TranslationValidationError(f"Corrupt or missing translated ZIP: {path}") from exc
    with archive:
        names = _safe_zip_members(archive)
        expected_members = set(expected_names) | {"translation_report.json"}
        if set(names) != expected_members:
            missing = sorted(expected_members.difference(names))
            extra = sorted(set(names).difference(expected_members))
            raise TranslationValidationError(
                f"Translated ZIP member mismatch; missing={missing}, extra={extra}"
            )
        _verify_zip_crc(archive)
        batches: list[BatchFile] = []
        for name in expected_names:
            records = _decode_jsonl(archive.read(name), name)
            if not records:
                raise TranslationValidationError(
                    f"Translated batch contains zero records: {name}"
                )
            batches.append(BatchFile(name, tuple(records)))
        report_document = _decode_json(
            archive.read("translation_report.json"), "translation_report.json"
        )

    return validate_translation_records(
        trusted,
        batches,
        translation_report=report_document,
        known_names=known_names,
        forbidden_name_variants=forbidden_name_variants,
        source_name_variants=source_name_variants,
        semantic_window=semantic_window,
        expected_batch_uids=expected_batch_uids,
        raise_on_error=raise_on_error,
    )


validate_translated_zip = load_and_validate_translated_zip


__all__ = [
    "DEFAULT_FORBIDDEN_NAME_VARIANTS",
    "DEFAULT_KNOWN_NAMES",
    "DEFAULT_SOURCE_NAME_VARIANTS",
    "MAX_TRANSLATED_ZIP_COMPRESSION_RATIO",
    "MAX_TRANSLATED_ZIP_MEMBER_BYTES",
    "MAX_TRANSLATED_ZIP_MEMBERS",
    "MAX_TRANSLATED_ZIP_TOTAL_BYTES",
    "TRANSLATION_RECORD_FIELDS",
    "TRANSLATION_REPORT_FIELDS",
    "TranslationValidationError",
    "TranslationValidationReport",
    "TranslationValidationResult",
    "ValidationIssue",
    "load_and_validate_translated_zip",
    "validate_translated_zip",
    "validate_translation_records",
]
