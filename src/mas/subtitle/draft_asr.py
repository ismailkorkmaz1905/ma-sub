import copy
import math
import re

from ..engine.forced_align import _alignment_model_text
from ..reliability import IntegrityError, digest


def reject_new_alignment_overlaps(original, prepared):
    prepared = copy.deepcopy(prepared)
    changed = {i for i, (old, new) in enumerate(zip(original, prepared))
               if old['start'] != new['start'] or old['end'] != new['end']}
    rejected = set()
    while changed:
        conflicts = set()
        order = sorted(range(len(prepared)), key=lambda i: prepared[i]['start'])
        for position, left in enumerate(order):
            for right in order[position + 1:]:
                if prepared[right]['start'] >= prepared[left]['end'] - 1e-9:
                    break
                old_overlap = min(original[left]['end'], original[right]['end']) - max(
                    original[left]['start'], original[right]['start'])
                if old_overlap <= 1e-9:
                    conflicts.update({left, right} & changed)
        if not conflicts:
            break
        for index in conflicts:
            prepared[index] = copy.deepcopy(original[index])
        changed -= conflicts
        rejected |= conflicts
    return prepared, [{'kind': 'ctc_new_neighbor_overlap_review', 'segment_index': i,
                       'used': 'original_asr'} for i in sorted(rejected)]


def apply_ctc_alignment(segments, aligned):
    if len(segments) != len(aligned):
        raise IntegrityError('CTC segment set is incomplete')
    prepared, issues = copy.deepcopy(segments), []
    accepted = 0
    for index, (segment, record) in enumerate(zip(segments, aligned)):
        if record['index'] != index or record['input_segment_sha256'] != digest(segment):
            raise IntegrityError('CTC segment differs from ASR input')
        tokens = re.findall(r'\S+', segment['text'])
        words = record['raw']['word_segments']
        reason = None
        if len(tokens) != len(words) or any(
                _alignment_model_text(token).casefold() != word['word'].casefold()
                for token, word in zip(tokens, words)):
            reason = 'ctc_token_coverage_review'
        else:
            previous = -1
            for word in words:
                values = [word.get(key) for key in ('start', 'end', 'score')]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                    reason = 'ctc_untimed_word_review'
                    break
                start, end, score = values
                if not (start >= previous and 0 < end - start <= 2.5 and .3 <= score <= 1):
                    reason = 'ctc_word_quality_review'
                    break
                # Padding arithmetic can leave a binary float error below 1 ns.
                if (start < record['coarse']['start'] - 1e-9
                        or end > record['coarse']['end'] + 1e-9):
                    reason = 'ctc_outside_audio_window_review'
                    break
                previous = end
        if reason:
            issues.append({'kind': reason, 'segment_index': index, 'used': 'original_asr'})
            continue
        prepared[index]['words'] = [
            {'word': (' ' if i else '') + token, 'start': word['start'],
             'end': word['end'], 'probability': word['score']}
            for i, (token, word) in enumerate(zip(tokens, words))]
        prepared[index]['start'] = words[0]['start']
        prepared[index]['end'] = words[-1]['end']
        accepted += 1
    prepared, overlap_issues = reject_new_alignment_overlaps(segments, prepared)
    issues.extend(overlap_issues)
    return prepared, issues, accepted - len(overlap_issues)


def coalesce_zero_duration_fragments(segments):
    prepared, issues = copy.deepcopy(segments), []
    for index, segment in enumerate(prepared):
        words = segment.get('words', [])
        if any(word['end'] < word['start'] for word in words):
            raise IntegrityError('negative ASR word duration')
        if not any(word['end'] == word['start'] for word in words):
            continue
        if not any(word['end'] > word['start'] for word in words):
            raise IntegrityError('ASR segment has no positive word interval; acoustic repair required')
        joined, pending = [], []
        for word_index, word in enumerate(words):
            if word['end'] == word['start']:
                issues.append({'kind': 'zero_duration_fragment_review',
                               'segment_index': index, 'word_index': word_index,
                               'original_word': copy.deepcopy(word)})
                if joined and word['start'] == joined[-1]['end']:
                    joined[-1]['word'] += word['word']
                else:
                    pending.append(word)
                continue
            if pending:
                if any(part['start'] != word['start'] for part in pending):
                    raise IntegrityError('zero duration fragments have no adjacent acoustic interval')
                word['word'] = ''.join(part['word'] for part in pending) + word['word']
                pending = []
            joined.append(word)
        if pending:
            raise IntegrityError('unresolved trailing ASR fragments')
        original = ''.join(w['word'] for w in segments[index]['words'])
        if original != ''.join(w['word'] for w in joined):
            raise IntegrityError('fragment coalescing changed transcript')
        segment['words'] = joined
    return prepared, issues
