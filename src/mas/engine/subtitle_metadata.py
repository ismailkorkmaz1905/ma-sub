"""Exact credit quarantine and non-destructive repetition diagnostics."""

import re
import unicodedata
from collections import defaultdict


def normalized_text(text):
    text = unicodedata.normalize('NFKC', str(text)).translate(str.maketrans({'İ': 'i', 'I': 'ı'}))
    text = ''.join(character for character in text if unicodedata.category(character) != 'Cf')
    return ' '.join(re.findall(r'[^\W_]+', text.casefold()))


def is_subtitle_credit(text):
    words = normalized_text(text).split()
    return (len(words) == 3 and words[0].replace('ı', 'i') in {'altyazi', 'takarir', 'subtitle', 'subtitles'}
            and words[1:] in (['m', 'k'], ['k', 'm']))


def repetition_warnings(records, *, text_key='text', uid_key='uid', minimum_count=5):
    groups = defaultdict(list)
    for record in records:
        text = normalized_text(record.get(text_key, ''))
        if text and type(record.get('start_ms')) is int and type(record.get('end_ms')) is int:
            groups[text].append(record)
    warnings = []
    for text, group in groups.items():
        if len(group) < minimum_count:
            continue
        start = min(record['start_ms'] for record in group)
        end = max(record['end_ms'] for record in group)
        warnings.append({
            'reason': 'repeated_subtitle_text', 'action': 'reported_without_text_change',
            'normalized_text': text, 'count': len(group),
            'uids': [record.get(uid_key) for record in group],
            'start_ms': start, 'end_ms': end, 'span_ms': end - start,
            'minute_bins': sorted({record['start_ms'] // 60000 for record in group}),
            'known_credit': is_subtitle_credit(text),
        })
    return warnings
