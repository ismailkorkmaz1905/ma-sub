"""Bounded speech-gap recovery. VAD/captions select audio, never supply dialogue."""
import contextlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from .delivery_align import AlignmentUnavailable, BoundedAligner

MAX_TARGET_MS = 10_000
MAX_JOBS = 192  # Every job is ordinary work; there is no mandatory-work exemption.
TOTAL_SECONDS = 180
GROUP_SECONDS = 20
MIN_GAP_MS = 230


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def _uncovered(start, end, covered):
    cursor = start
    for a, b in covered:
        if b <= cursor:
            continue
        if a >= end:
            break
        if a > cursor:
            yield cursor, min(a, end)
        cursor = max(cursor, b)
        if cursor >= end:
            return
    if cursor < end:
        yield cursor, end


def plan_gaps(cues, vad_regions, captions, segments, duration_ms):
    covered = _merged((c['start_ms'], c['end_ms']) for c in cues)
    hints = []
    for region in vad_regions:
        hints.append((region['start_ms'], region['end_ms']))
    for caption in captions:
        text = str(caption.get('text', '')).strip()
        if text and not re.fullmatch(r'\[.*\]|\(.*\)', text):
            hints.append((caption['start_ms'], caption['end_ms']))
    for segment in segments:
        if str(segment.get('text', '')).strip():
            hints.append((segment['start_ms'], segment['end_ms']))
    gaps = []
    for a, b in _merged((max(0, a), min(duration_ms, b)) for a, b in hints):
        for start, end in _uncovered(a, b, covered):
            if end - start >= MIN_GAP_MS:
                gaps.extend({'start_ms': s, 'end_ms': min(s + MAX_TARGET_MS, end)}
                            for s in range(start, end, MAX_TARGET_MS))
    return gaps


def model_binding(series):
    from . import raw_asr, primary_checkpoint
    directory = primary_checkpoint.resolve_model(series['whisper_model'])
    return {'model': primary_checkpoint.model_identity(directory),
            'producer': raw_asr._raw_asr_producer_identity()}


class BoundedRescuer(BoundedAligner):
    def __init__(self, audio_path, log_path, *, model_name, expected_model, startup_seconds):
        super().__init__(audio_path, log_path, startup_seconds=startup_seconds,
                         command=[sys.executable, '-u', '-m', 'mas.engine.delivery_coverage',
                                  str(audio_path), model_name])
        if self.ready.get('model') != expected_model:
            self.close()
            raise ValueError('Rescue model differs from the authenticated primary model')


def _valid_rescue(result, target):
    from ..delivery_first import source_cues
    if not isinstance(result, dict) or result.get('status') != 'rescued':
        return []
    segments = result.get('segments', [])
    if not isinstance(segments, list):
        return []
    usable = []
    for segment in segments:
        if (not isinstance(segment, dict) or type(segment.get('start_ms')) is not int
                or type(segment.get('end_ms')) is not int
                or not target['start_ms'] <= segment['start_ms'] < segment['end_ms'] <= target['end_ms']):
            continue
        usable.append(segment)
    cues, _ = source_cues(usable, target['end_ms'])
    for cue in cues:
        cue['uid'] = 'rescue-' + target['uid'] + '-' + cue['uid']
        cue['timing_source'] = 'rescue_asr_interval'
    return cues


def recover_gaps(cues, segments, vad_regions, captions_path, audio_path, child, binding, *,
                 remaining, model_name, reserve_seconds=1800, session_factory=None):
    from ..delivery_first import read_signed, write_signed
    from ..reliability import digest
    from ..progress import mark_work_progress
    from .transcribe import load_vtt_captions
    session_factory = session_factory or BoundedRescuer
    child = Path(child)
    duration_ms = binding['duration_ms']
    captions = load_vtt_captions(captions_path)
    plan = plan_gaps(cues, vad_regions, captions, segments, duration_ms)
    root = child / 'work/delivery-rescue'
    root.mkdir(parents=True, exist_ok=True)
    budget_path = child.parent.parent / 'work/delivery-rescue-budget.json'
    budget_identity = {'episode': binding['episode'], 'plan_sha256': binding['plan_sha256'],
                       'limit_seconds': TOTAL_SECONDS, 'max_jobs': MAX_JOBS}
    budget = (read_signed(budget_path, 'rescue-budget') if budget_path.exists()
              else {**budget_identity, 'spent_seconds': 0.0, 'jobs': 0})
    if (any(budget.get(k) != v for k, v in budget_identity.items())
            or type(budget.get('spent_seconds')) not in (int, float)
            or not math.isfinite(budget['spent_seconds']) or not 0 <= budget['spent_seconds'] <= TOTAL_SECONDS
            or type(budget.get('jobs')) is not int or not 0 <= budget['jobs'] <= MAX_JOBS):
        raise ValueError('Rescue budget identity changed')
    basis = {'binding': binding, 'targets_sha256': digest(plan)}
    result_cues, warnings = list(cues), []
    session = None
    failures = 0
    try:
        for index, span in enumerate(plan):
            target = {**span, 'uid': digest({'basis': basis, 'span': span})}
            path = root / (target['uid'] + '.json')
            if path.exists():
                saved = read_signed(path, 'rescue-group')
                if saved.get('basis') != basis or saved.get('target') != target:
                    raise ValueError('Rescue result source/target binding changed')
                result = saved.get('result') or {'status': 'unresolved', 'reason': 'interrupted_rescue'}
            else:
                result = {'status': 'unresolved', 'reason': 'rescue_budget_or_delivery_reserve'}
                allowance = min(GROUP_SECONDS, TOTAL_SECONDS - budget['spent_seconds'],
                                max(0, remaining() - reserve_seconds))
                if allowance >= .1 and budget['jobs'] < MAX_JOBS and failures < 2:
                    if session is None:
                        startup = min(60, TOTAL_SECONDS - budget['spent_seconds'],
                                      max(0, remaining() - reserve_seconds))
                        budget['spent_seconds'] += startup
                        write_signed(budget_path, budget, 'rescue-budget')
                        started = time.monotonic()
                        try:
                            session = session_factory(audio_path, root / 'worker.log', model_name=model_name,
                                expected_model=binding['primary_producer']['model'], startup_seconds=startup)
                        except AlignmentUnavailable as exc:
                            result['reason'] = str(exc)
                            failures += 1
                        finally:
                            budget['spent_seconds'] -= max(0, startup - (time.monotonic() - started))
                            write_signed(budget_path, budget, 'rescue-budget')
                    allowance = min(GROUP_SECONDS, TOTAL_SECONDS - budget['spent_seconds'],
                                    max(0, remaining() - reserve_seconds))
                    if session is not None and allowance >= .1:
                        budget['jobs'] += 1
                        budget['spent_seconds'] += allowance
                        write_signed(budget_path, budget, 'rescue-budget')
                        write_signed(path, {'basis': basis, 'target': target, 'result': None}, 'rescue-group')
                        started = time.monotonic()
                        try:
                            result = session.align(target, allowance)
                        except AlignmentUnavailable as exc:
                            result = {'status': 'unresolved', 'reason': str(exc)}
                            session.close()
                            session = None
                            failures += 1
                        finally:
                            budget['spent_seconds'] -= max(0, allowance - (time.monotonic() - started))
                            write_signed(budget_path, budget, 'rescue-budget')
                write_signed(path, {'basis': basis, 'target': target, 'result': result}, 'rescue-group')
            recovered = _valid_rescue(result, target)
            result_cues.extend(recovered)
            recovered_intervals = _merged((c['start_ms'], c['end_ms']) for c in recovered)
            residual = [(a, b) for a, b in _uncovered(span['start_ms'], span['end_ms'], recovered_intervals)
                        if b - a >= MIN_GAP_MS]
            warnings.append({'reason': 'speech_gap_recovered' if recovered else 'speech_gap_unresolved',
                'start_ms': span['start_ms'], 'end_ms': span['end_ms'], 'target_uid': target['uid'],
                'recovered_cues': len(recovered), 'residual_intervals_ms': residual,
                'detail': result.get('reason', 'decoded_audio'), 'action': 'reported_not_blocked'})
            mark_work_progress('delivery_speech_recovery', completed=index + 1)
    finally:
        if session is not None:
            session.close()
    warnings.append({'reason': 'speech_coverage_not_acoustically_certified',
                     'action': 'reported_not_blocked', 'hint_count': len(plan)})
    return result_cues, warnings


def _worker(audio_path, model_name):
    def emit(value):
        print('MAS_ALIGN ' + json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from . import raw_asr as raw, primary_checkpoint as checkpoint
            settings = raw.RawASRV2Config(model_name=model_name)
            model_dir = checkpoint.resolve_model(model_name)
            model_id = checkpoint.model_identity(model_dir)
            WhisperModel, ctranslate2 = raw._import_whisper()
            device, compute = raw._select_device(settings.transcription_config(), ctranslate2)
            if device != 'cuda':
                raise ValueError('No CUDA for recovery')
            model = WhisperModel(str(model_dir), device='cuda', compute_type=compute, num_workers=1)
        emit({'event': 'ready', 'model': model_id})
    except Exception as exc:
        emit({'event': 'unavailable', 'reason': type(exc).__name__})
        return
    with tempfile.TemporaryDirectory(prefix='mas-delivery-rescue-') as directory:
        clip = Path(directory) / 'target.wav'
        for line in sys.stdin:
            request = json.loads(line)
            answer = {'uid': request['uid'], 'status': 'unresolved', 'reason': 'rescue_rejected'}
            try:
                start, end = request['start_ms'], request['end_ms']
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= start + MAX_TARGET_MS:
                    raise ValueError('Unbounded rescue request')
                with contextlib.redirect_stdout(sys.stderr):
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-accurate_seek',
                        '-ss', f'{start / 1000:.3f}', '-i', audio_path, '-t', f'{(end - start) / 1000:.3f}',
                        '-ac', '1', '-ar', '16000', str(clip)], stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE, timeout=GROUP_SECONDS, check=True)
                    iterator, _ = model.transcribe(str(clip), language='tr', task='transcribe',
                        word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
                        temperature=0.0, beam_size=5)
                    segments, _ = raw.consume_coarse_segments(iterator, offset_ms=start, source='delivery-rescue')
                answer.update(status='rescued', segments=segments)
            except Exception as exc:
                answer['reason'] = type(exc).__name__
            emit(answer)


if __name__ == '__main__':
    _worker(sys.argv[1], sys.argv[2])
