"""Offline closure planning and a persisted budget for authorized CTC recovery."""

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path

from ..reliability import atomic_json, digest


class RecoveryPlanError(ValueError):
    pass


def valid_request(transcript, request_uids):
    if (not isinstance(request_uids, list) or not request_uids
            or any(not isinstance(uid, str) or not uid for uid in request_uids)
            or len(set(request_uids)) != len(request_uids)
            or not isinstance(transcript, list) or len(transcript) != 1
            or not isinstance(transcript[0], dict)):
        return False
    request = transcript[0]
    return (isinstance(request.get('text'), str) and bool(request['text'].strip())
            and all(type(request.get(key)) in (int, float)
                    and math.isfinite(request[key]) for key in ('start', 'end'))
            and 0 <= request['start'] < request['end'])


def build_recovery_plan(source, target_uids, *, max_new_ctc_calls):
    from .forced_align import (MAX_CONFLICT_ALIGNMENT_CALLS,
                               MAX_RECOVERY_CONTEXT_GAP_MS,
                               validate_coarse_segments)
    source = validate_coarse_segments(source)
    ordered = [item['utterance_uid'] for item in source]
    if (not isinstance(target_uids, list) or not target_uids
            or any(not isinstance(uid, str) for uid in target_uids)
            or target_uids != [uid for uid in ordered if uid in target_uids]):
        raise RecoveryPlanError('closure roots must be exact ordered unique source UIDs')
    if (type(max_new_ctc_calls) is not int
            or not 1 <= max_new_ctc_calls <= len(source) * MAX_CONFLICT_ALIGNMENT_CALLS):
        raise RecoveryPlanError('closure needs an explicit bounded new-CTC-call allowance')

    # A context expansion can repeatedly reach the next cue. Freeze the full
    # connected region, not one more radius or one more discovered component.
    # Overlapping source windows also connect regions across a coarse-time gap.
    components = []
    group = []
    window_end = -1
    previous = None
    for item in source:
        if (previous is not None
                and item['coarse_start_ms'] - previous['coarse_end_ms']
                > MAX_RECOVERY_CONTEXT_GAP_MS
                and item['start_ms'] > window_end):
            components.append(group)
            group = []
            window_end = -1
        group.append(item['utterance_uid'])
        window_end = max(window_end, item['end_ms'])
        previous = item
    components.append(group)
    roots = set(target_uids)
    return {
        'format': 'mas-alignment-closure-1',
        'source_sha256': digest(source),
        'root_uids': list(target_uids),
        'components': [group for group in components if roots.intersection(group)],
        'max_new_ctc_calls': max_new_ctc_calls,
    }


def validate_recovery_plan(plan, source, target_uids):
    if not isinstance(plan, dict):
        raise RecoveryPlanError('closure plan must be an object')
    expected = build_recovery_plan(
        source, target_uids, max_new_ctc_calls=plan.get('max_new_ctc_calls'))
    if plan != expected:
        raise RecoveryPlanError('closure plan differs from the exact source dependency closure')
    return expected


def request_matches_plan(transcript, request_uids, source, plan):
    from .forced_align import _alignment_model_text
    if not valid_request(transcript, request_uids):
        return False
    requested = set(request_uids)
    component = next((group for group in plan['components']
                      if requested <= set(group)), None)
    if component is None:
        return False
    allowed = set(component)
    group = [item for item in source if item['utterance_uid'] in allowed]
    request = transcript[0]
    text = request['text']

    def run_admissible(run):
        # UIDs, model text and audio window must all describe the same source
        # selection. Matching them independently, or only against the whole
        # closure envelope, would let one cue's text carry another cue's audio.
        return (bool(run)
                and set(request_uids) <= {item['utterance_uid'] for item in run}
                and ' '.join(_alignment_model_text(item['text'])
                             for item in run) == text
                and request['start'] >= min(item['start_ms'] for item in run) / 1000
                and request['end'] <= max(item['end_ms'] for item in run) / 1000)

    # A conflict component is connected by word overlap, so its UIDs need not be
    # contiguous in source order; such a joint call carries exactly its own
    # source text, in source order.
    if run_admissible([item for item in group
                       if item['utterance_uid'] in requested]):
        return True
    # Exact, contiguous source text stays admissible: the request UIDs are the
    # replaceable targets, the run supplies the acoustic context. The approved
    # closure supplies context only; word/score/drift validators still run.
    # Repeated dialogue means several runs can carry the same text, so a
    # rejected candidate must not reject the request.
    for start in range(len(group)):
        parts = []
        for index in range(start, len(group)):
            parts.append(_alignment_model_text(group[index]['text']))
            candidate = ' '.join(parts)
            if candidate == text:
                if run_admissible(group[start:index + 1]):
                    return True
                break
            if len(candidate) >= len(text) or not text.startswith(candidate):
                break
    return False


class RecoveryCallBudget:
    """Charge before a new call; completed raw-cache hits do not consume budget."""

    def __init__(self, checkpoint_dir, scope):
        value = os.getenv('MAS_RAW_ASR_AUTH_KEY', '')
        if not re.fullmatch(r'[0-9a-f]{64}', value):
            raise RecoveryPlanError('authenticated closure recovery requires MAS_RAW_ASR_AUTH_KEY')
        self.key = bytes.fromhex(value)
        self.binding = digest(scope)
        self.limit = scope['recovery_plan']['max_new_ctc_calls']
        self.path = (Path(checkpoint_dir) / 'recovery-plans' / self.binding
                     / 'call-budget.json')
        self._read()

    def _tag(self, body):
        return hmac.new(self.key, b'ma-sub/alignment-closure-calls/v1\0'
                        + digest(body).encode('ascii'), hashlib.sha256).hexdigest()

    def _read(self):
        body = {'format': 'mas-alignment-closure-calls-1',
                'scope_sha256': self.binding, 'attempts': []}
        if self.path.is_symlink():
            raise RecoveryPlanError('closure call budget must be a regular file')
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            body = saved.get('data')
            tag = saved.get('auth_tag')
            if (not isinstance(body, dict) or not isinstance(tag, str)
                    or not hmac.compare_digest(tag, self._tag(body))
                    or set(body) != {'format', 'scope_sha256', 'attempts'}
                    or body['format'] != 'mas-alignment-closure-calls-1'
                    or body['scope_sha256'] != self.binding
                    or not isinstance(body['attempts'], list)
                    or len(body['attempts']) > self.limit
                    or any(not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{64}', key)
                           for key in body['attempts'])):
                raise RecoveryPlanError('closure call budget authentication or identity changed')
        return body

    def reserve(self, request_sha256):
        if not isinstance(request_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', request_sha256):
            raise RecoveryPlanError('invalid closure request identity')
        body = self._read()
        if len(body['attempts']) >= self.limit:
            raise RecoveryPlanError('authorized closure new-CTC-call allowance exhausted')
        # A crash after this write conservatively consumes one attempt. Repeating
        # an interrupted call requires another remaining slot, never a reset.
        body['attempts'].append(request_sha256)
        atomic_json(self.path, {'data': body, 'auth_tag': self._tag(body)})
