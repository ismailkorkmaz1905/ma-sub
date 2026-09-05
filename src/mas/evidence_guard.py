"""Versioned EP12 evidence checks, deliberately separate from legacy V2 gates.

No text is invented from report summaries. Timing changes cannot modify text.
Cross-speaker overlap is classified, not merged, and requires explicit speaker
identity to be accepted without review. These checks do not rewrite SRT cues.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, fields
from typing import Mapping, Sequence
from .reliability import IntegrityError


def interval(start: int, end: int) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise IntegrityError('interval must have integer milliseconds and end > start >= 0')
    return start, end


def union(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(interval(a, b) for a, b in intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract(target: tuple[int, int], covered: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    start, end = interval(*target)
    result = []
    for a, b in union(covered):
        if b <= start or a >= end:
            continue
        if a > start:
            result.append((start, min(a, end)))
        start = max(start, b)
        if start >= end:
            break
    if start < end:
        result.append((start, end))
    return result


@dataclass(frozen=True)
class ReviewWindow:
    clip_start_ms: int
    clip_end_ms: int
    target_start_ms: int
    target_end_ms: int
    evidence_audio_sha256: str

    def __post_init__(self):
        interval(self.clip_start_ms, self.clip_end_ms)
        interval(self.target_start_ms, self.target_end_ms)
        if not (self.clip_start_ms <= self.target_start_ms < self.target_end_ms <= self.clip_end_ms):
            raise IntegrityError('target must be inside the context clip')
        if not re.fullmatch('[0-9a-f]{64}', self.evidence_audio_sha256):
            raise IntegrityError('review evidence must have a SHA-256')

    def missing_speech(self, speech: Sequence[tuple[int, int]],
                       represented: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Only measured speech inside the target needs word coverage.

        Empty speech is NOT a non-dialogue decision. The caller must retain an
        explicit acoustic decision before clearing a review candidate.
        """
        missing = []
        for a, b in union(speech):
            a, b = max(a, self.target_start_ms), min(b, self.target_end_ms)
            if a < b:
                missing.extend(subtract((a, b), represented))
        return union(missing)


@dataclass(frozen=True)
class TimingOverride:
    input_sha256: str
    utterance_uid: str
    expected_old_text: str
    expected_start_ms: int
    expected_end_ms: int
    start_ms: int
    end_ms: int

    @classmethod
    def from_dict(cls, value: Mapping):
        allowed = {f.name for f in fields(cls)}
        if set(value) != allowed:
            raise IntegrityError(f'timing override keys differ: {sorted(set(value) ^ allowed)}')
        return cls(**value)

    def apply(self, record: Mapping, *, input_sha256: str) -> dict:
        interval(self.expected_start_ms, self.expected_end_ms)
        interval(self.start_ms, self.end_ms)
        interval(record.get('start_ms'), record.get('end_ms'))
        if not re.fullmatch('[0-9a-f]{64}', self.input_sha256):
            raise IntegrityError('override input_sha256 is invalid')
        expected = (self.input_sha256, self.utterance_uid, self.expected_old_text,
                    self.expected_start_ms, self.expected_end_ms)
        actual = (input_sha256, record.get('utterance_uid'), record.get('tr_corrected'),
                  record.get('start_ms'), record.get('end_ms'))
        if actual != expected:
            raise IntegrityError(f'stale timing override precondition: {self.utterance_uid}')
        result = copy.deepcopy(dict(record))
        result.update(start_ms=self.start_ms, end_ms=self.end_ms)
        return result


def semantic_shrink(old: str, new: str) -> bool:
    """Review alarm only, not a machine judgment that a correction is wrong."""
    def words(value):
        return re.findall(r'[^\W_]+', value.casefold(), re.UNICODE)
    before, after = words(old), words(new)
    return len(before) >= 6 and (len(after) <= 1 or len(after) / len(before) < 0.35)


def classify_word_lanes(words: Sequence[Mapping]) -> dict:
    """Reject invalid/same-utterance timing; classify other speaker overlap.

    Evidence order is validated before sorting. A missing speaker ID does not
    manufacture a new speaker. Cross-utterance unknown overlap stays review.
    """
    last_by_uid = {}
    checked = []
    for word in words:
        uid = word.get('utterance_uid')
        if not isinstance(uid, str) or not uid:
            raise IntegrityError('word missing utterance_uid')
        start, end = interval(word.get('start_ms'), word.get('end_ms'))
        if uid in last_by_uid and start < last_by_uid[uid]:
            raise IntegrityError(f'non-monotonic/overlapping words within UID {uid}')
        last_by_uid[uid] = end
        checked.append(dict(word))
    active, accepted, review = [], [], []
    for word in sorted(checked, key=lambda w: (w['start_ms'], w['end_ms'])):
        active = [a for a in active if a['end_ms'] > word['start_ms']]
        for prev in active:
            speaker_a, speaker_b = prev.get('speaker_id'), word.get('speaker_id')
            pair = {'first_uid': prev['utterance_uid'], 'second_uid': word['utterance_uid'],
                    'start_ms': word['start_ms'],
                    'end_ms': min(prev['end_ms'], word['end_ms'])}
            if speaker_a and speaker_b and speaker_a != speaker_b:
                accepted.append(dict(pair, reason='cross_speaker_overlap'))
            else:
                review.append(dict(pair, reason='same_speaker_overlap' if speaker_a and speaker_b
                                   else 'unknown_speaker_overlap'))
        active.append(word)
    return {'accepted_overlaps': accepted, 'review_required': review,
            'word_count': len(words), 'text_mutated': False}
