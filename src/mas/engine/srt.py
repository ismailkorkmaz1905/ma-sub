"""Strict, deterministic UTF-8 SubRip (SRT) support.

The immutable schema owns cue order and timing.  This module only lays out text;
it never moves text between block UIDs and never derives timing from a
translation record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>\d{2}):(?P<seconds>\d{2})"
    r",(?P<millis>\d{3})$"
)
_TIMING_LINE_RE = re.compile(
    r"^(?P<start>\d{2,}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2,}:\d{2}:\d{2},\d{3})$"
)
_TAG_RE = re.compile(r"<[^>]*>|\{\\[^}]*\}")


class SRTError(ValueError):
    """Raised when SRT data is malformed or cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class SubtitleEntry:
    """One immutable SRT cue."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise SRTError("Subtitle index must be an integer")
        if self.index < 1:
            raise SRTError("Subtitle index must be at least 1")
        for label, value in (("start_ms", self.start_ms), ("end_ms", self.end_ms)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SRTError(f"{label} must be an integer number of milliseconds")
            if value < 0:
                raise SRTError(f"{label} cannot be negative")
        if self.end_ms <= self.start_ms:
            raise SRTError(
                f"Subtitle {self.index} must end after it starts "
                f"({self.start_ms} >= {self.end_ms})"
            )
        _validate_text(self.text, self.index)


# Compatibility alias used by a few subtitle libraries and older notebooks.
SRTEntry = SubtitleEntry


def format_timestamp(milliseconds: int) -> str:
    """Format non-negative integer milliseconds as ``HH:MM:SS,mmm``."""

    if isinstance(milliseconds, bool) or not isinstance(milliseconds, int):
        raise SRTError("Timestamp must be an integer number of milliseconds")
    if milliseconds < 0:
        raise SRTError("Timestamp cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_timestamp(value: str) -> int:
    """Parse a strict SubRip timestamp into milliseconds."""

    if not isinstance(value, str):
        raise SRTError("Timestamp must be text")
    match = _TIMESTAMP_RE.fullmatch(value.strip())
    if not match:
        raise SRTError(f"Invalid SRT timestamp: {value!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = int(match.group("millis"))
    if minutes > 59 or seconds > 59:
        raise SRTError(f"Invalid SRT timestamp: {value!r}")
    return (((hours * 60) + minutes) * 60 + seconds) * 1_000 + millis


def visible_length(text: str) -> int:
    """Return the visible character count, ignoring common subtitle tags."""

    return len(_TAG_RE.sub("", text))


def _clean_line(value: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", value).strip()


def _validate_text(text: str, index: int | None = None) -> None:
    label = f"Subtitle {index}" if index is not None else "Subtitle"
    if not isinstance(text, str):
        raise SRTError(f"{label} text must be a string")
    if "\x00" in text:
        raise SRTError(f"{label} text contains a NUL character")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise SRTError(f"{label} text contains an unsupported control character")
    if not text.strip():
        raise SRTError(f"{label} text is empty")


def _best_two_line_split(words: Sequence[str], target: int, hard_limit: int) -> tuple[str, str] | None:
    """Choose a readable, deterministic two-line word-boundary split."""

    candidates: list[tuple[tuple[int, int, int, int], str, str]] = []
    for position in range(1, len(words)):
        first = " ".join(words[:position])
        second = " ".join(words[position:])
        first_len = visible_length(first)
        second_len = visible_length(second)
        if first_len > hard_limit or second_len > hard_limit:
            continue
        # First minimize hard target overflow, then balance the two lines.  A
        # slightly longer first line is preferred when all else is equal.
        score = (
            max(0, first_len - target) + max(0, second_len - target),
            max(first_len, second_len),
            abs(first_len - second_len),
            -first_len,
        )
        candidates.append((score, first, second))
    if not candidates:
        return None
    _, first, second = min(candidates, key=lambda item: item[0])
    return first, second


def wrap_text(
    text: str,
    target: int = 42,
    max_lines: int = 2,
    *,
    hard_limit: int = 84,
) -> str:
    """Lay out subtitle text without changing word order.

    ``target`` is a soft target. ``hard_limit`` and ``max_lines`` are hard
    limits. Explicit two-line dialogue is preserved. Text which cannot fit at
    word boundaries fails rather than being truncated or silently moved.
    """

    _validate_text(text)
    if target < 1 or hard_limit < 1 or max_lines < 1:
        raise SRTError("Line limits must be positive integers")
    if target > hard_limit:
        raise SRTError("The target width cannot exceed the hard line limit")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    explicit_lines = [_clean_line(line) for line in normalized.split("\n")]
    if any(not line for line in explicit_lines):
        raise SRTError("Subtitle text contains an empty visible line")
    if len(explicit_lines) > max_lines:
        raise SRTError(
            f"Subtitle has {len(explicit_lines)} lines; maximum is {max_lines}"
        )
    if len(explicit_lines) > 1:
        for line in explicit_lines:
            if visible_length(line) > hard_limit:
                raise SRTError(
                    f"Explicit subtitle line is {visible_length(line)} characters; "
                    f"maximum is {hard_limit}"
                )
        return "\n".join(explicit_lines)

    one_line = explicit_lines[0]
    if visible_length(one_line) <= target or max_lines == 1:
        if visible_length(one_line) > hard_limit:
            raise SRTError(
                f"Subtitle line is {visible_length(one_line)} characters; "
                f"maximum is {hard_limit}"
            )
        return one_line

    words = one_line.split(" ")
    if any(visible_length(word) > hard_limit for word in words):
        raise SRTError("Subtitle contains a word longer than the hard line limit")
    if max_lines != 2:
        # This workflow intentionally supports at most two visible lines.
        raise SRTError("Only one- or two-line subtitle layout is supported")
    split = _best_two_line_split(words, target, hard_limit)
    if split is None:
        raise SRTError(
            f"Subtitle cannot fit within {max_lines} lines of {hard_limit} characters"
        )
    return "\n".join(split)


def _coerce_int(value: Any, field: str, uid: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SRTError(f"Block {uid}: {field} must be an integer")
    return value


def build_entries(
    blocks: Sequence[Mapping[str, Any]],
    translations_by_uid: Mapping[str, Mapping[str, Any]],
    language: str = "tr",
    *,
    target: int = 42,
    max_lines: int = 2,
    hard_limit: int = 84,
) -> list[SubtitleEntry]:
    """Build entries from immutable schema blocks and UID-keyed translations.

    The function requires exact UID-set equality and contiguous block indexes.
    Translation record order and any echoed timing fields are ignored.
    """

    language_key = {"tr": "tr_final", "id": "id_final"}.get(language.lower())
    if language_key is None:
        raise SRTError("language must be 'tr' or 'id'")
    if not isinstance(translations_by_uid, Mapping):
        raise SRTError("translations_by_uid must be a block_uid-keyed mapping")

    expected_uids: list[str] = []
    seen: set[str] = set()
    for position, block in enumerate(blocks, start=1):
        uid = block.get("block_uid")
        if not isinstance(uid, str) or not uid:
            raise SRTError(f"Schema block at position {position} has no valid block_uid")
        if uid in seen:
            raise SRTError(f"Duplicate schema block_uid: {uid}")
        seen.add(uid)
        expected_uids.append(uid)
    actual_uids = set(translations_by_uid)
    missing = [uid for uid in expected_uids if uid not in actual_uids]
    extra = sorted(actual_uids - seen)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing UIDs: {', '.join(missing[:10])}")
        if extra:
            details.append(f"extra UIDs: {', '.join(extra[:10])}")
        raise SRTError("Translation UID set mismatch (" + "; ".join(details) + ")")

    entries: list[SubtitleEntry] = []
    for position, block in enumerate(blocks, start=1):
        uid = str(block["block_uid"])
        block_index = _coerce_int(block.get("block_index"), "block_index", uid)
        if block_index != position:
            raise SRTError(
                f"Block {uid}: expected contiguous block_index {position}, got {block_index}"
            )
        start_ms = _coerce_int(block.get("start_ms"), "start_ms", uid)
        end_ms = _coerce_int(block.get("end_ms"), "end_ms", uid)
        record = translations_by_uid[uid]
        if not isinstance(record, Mapping):
            raise SRTError(f"Translation for block {uid} is not an object")
        echoed_uid = record.get("block_uid", uid)
        if echoed_uid != uid:
            raise SRTError(
                f"Translation mapping key {uid} carries block_uid {echoed_uid!r}"
            )
        value = record.get(language_key)
        if not isinstance(value, str) or not value.strip():
            raise SRTError(f"Block {uid}: {language_key} is empty or missing")
        laid_out = wrap_text(
            value,
            target=target,
            max_lines=max_lines,
            hard_limit=hard_limit,
        )
        entries.append(SubtitleEntry(block_index, start_ms, end_ms, laid_out))
    return entries


def render_srt(entries: Iterable[SubtitleEntry], *, newline: str = "\r\n") -> str:
    """Serialize entries to canonical SRT text."""

    if newline not in {"\n", "\r\n"}:
        raise SRTError("SRT newline must be LF or CRLF")
    materialized = list(entries)
    parts: list[str] = []
    previous_end: int | None = None
    for expected_index, entry in enumerate(materialized, start=1):
        if not isinstance(entry, SubtitleEntry):
            raise SRTError(f"Entry {expected_index} is not a SubtitleEntry")
        if entry.index != expected_index:
            raise SRTError(
                f"Non-contiguous SRT index: expected {expected_index}, got {entry.index}"
            )
        # Overlaps belong to QA, not serialization.  They remain detectable and
        # are not silently adjusted here.
        previous_end = entry.end_ms if previous_end is None else max(previous_end, entry.end_ms)
        text = entry.text.replace("\r\n", "\n").replace("\r", "\n")
        block = newline.join(
            (
                str(entry.index),
                f"{format_timestamp(entry.start_ms)} --> {format_timestamp(entry.end_ms)}",
                *text.split("\n"),
            )
        )
        parts.append(block)
    if not parts:
        return ""
    return (newline + newline).join(parts) + newline + newline


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_srt(path: str | os.PathLike[str], entries: Iterable[SubtitleEntry]) -> Path:
    """Atomically write canonical UTF-8 SRT and verify semantic round-trip."""

    destination = Path(path)
    materialized = list(entries)
    payload = render_srt(materialized).encode("utf-8", errors="strict")
    # Validate bytes before exposing a final-looking filename.
    reparsed = parse_srt_text(payload.decode("utf-8", errors="strict"))
    _assert_entries_equal(reparsed, materialized)
    _atomic_write_bytes(destination, payload)
    assert_srt_roundtrip(destination, materialized)
    return destination


def parse_srt_text(text: str, *, strict_indexes: bool = True) -> list[SubtitleEntry]:
    """Parse strict SRT text without repairing malformed structure."""

    if not isinstance(text, str):
        raise SRTError("SRT input must be text")
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    if "\x00" in normalized:
        raise SRTError("SRT contains a NUL character")
    chunks = re.split(r"\n[ \t]*\n", normalized.strip("\n"))
    entries: list[SubtitleEntry] = []
    seen_indexes: set[int] = set()
    for block_number, chunk in enumerate(chunks, start=1):
        lines = chunk.split("\n")
        if len(lines) < 3:
            raise SRTError(f"SRT block {block_number} has no visible text")
        index_text = lines[0].strip()
        if not re.fullmatch(r"\d+", index_text):
            raise SRTError(f"SRT block {block_number} has invalid index {lines[0]!r}")
        index = int(index_text)
        if index in seen_indexes:
            raise SRTError(f"Duplicate SRT index: {index}")
        seen_indexes.add(index)
        if strict_indexes and index != block_number:
            raise SRTError(
                f"Non-contiguous SRT index at block {block_number}: got {index}"
            )
        timing_match = _TIMING_LINE_RE.fullmatch(lines[1].strip())
        if not timing_match:
            raise SRTError(f"SRT block {index} has invalid timing line {lines[1]!r}")
        start_ms = parse_timestamp(timing_match.group("start"))
        end_ms = parse_timestamp(timing_match.group("end"))
        # Preserve subtitle payload exactly. Whitespace normalization belongs to
        # ``wrap_text`` before generation, never to a parser used for mux
        # round-trip equality.
        text_lines = lines[2:]
        if any(not line.strip() for line in text_lines):
            raise SRTError(f"SRT block {index} contains an empty visible line")
        entries.append(SubtitleEntry(index, start_ms, end_ms, "\n".join(text_lines)))
    return entries


def parse_srt(path: str | os.PathLike[str]) -> list[SubtitleEntry]:
    """Read an SRT file as strict UTF-8 (an optional UTF-8 BOM is accepted)."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise SRTError(f"Cannot read SRT file {source}: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise SRTError(f"SRT file is not valid UTF-8: {source}") from exc
    return parse_srt_text(text)


def _assert_entries_equal(
    actual: Sequence[SubtitleEntry], expected: Sequence[SubtitleEntry]
) -> None:
    if len(actual) != len(expected):
        raise SRTError(
            f"SRT round-trip block count differs: expected {len(expected)}, got {len(actual)}"
        )
    for position, (actual_entry, expected_entry) in enumerate(
        zip(actual, expected), start=1
    ):
        if actual_entry != expected_entry:
            raise SRTError(
                "SRT round-trip mismatch at block "
                f"{position}: expected {expected_entry!r}, got {actual_entry!r}"
            )


def assert_srt_roundtrip(
    path: str | os.PathLike[str], expected: Sequence[SubtitleEntry]
) -> None:
    """Require exact index, millisecond timing, and text equality after parsing."""

    _assert_entries_equal(parse_srt(path), list(expected))


__all__ = [
    "SRTEntry",
    "SRTError",
    "SubtitleEntry",
    "assert_srt_roundtrip",
    "build_entries",
    "format_timestamp",
    "parse_srt",
    "parse_srt_text",
    "parse_timestamp",
    "render_srt",
    "visible_length",
    "wrap_text",
    "write_srt",
]
