"""Bounded, optional CTC refinement. Failures retain the source cue interval."""
import contextlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


class AlignmentUnavailable(RuntimeError):
    pass


class BoundedAligner:
    def __init__(self, audio_path, log_path, startup_seconds=90, command=None):
        self.events = queue.Queue()
        self.log = Path(log_path).open('ab')
        try:
            self.process = subprocess.Popen(
                command or [sys.executable, '-u', '-m', 'mas.engine.delivery_align', str(audio_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding='utf-8', start_new_session=(os.name != 'nt'))
        except OSError as exc:
            self.log.close()
            raise AlignmentUnavailable('alignment_worker_unavailable') from exc
        def consume():
            try:
                for line in self.process.stdout:
                    if line.startswith('MAS_ALIGN '):
                        self.events.put(json.loads(line[len('MAS_ALIGN '):]))
            except Exception:
                pass
            finally:
                self.events.put({'event': 'closed'})
        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        try:
            self.ready = self._receive(startup_seconds)
            if self.ready.get('event') != 'ready':
                raise AlignmentUnavailable('alignment_model_unavailable')
        except BaseException:
            self.close()
            raise

    def _receive(self, seconds):
        try:
            return self.events.get(timeout=max(0.001, seconds))
        except queue.Empty:
            raise AlignmentUnavailable('alignment_timeout') from None

    def align(self, cue, seconds):
        try:
            self.process.stdin.write(json.dumps(cue, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            result = self._receive(seconds)
        except (OSError, ValueError) as exc:
            raise AlignmentUnavailable('alignment_worker_lost') from exc
        if result.get('uid') != cue['uid']:
            raise AlignmentUnavailable('alignment_worker_lost')
        return result

    def close(self):
        process = self.process
        if process.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, timeout=10, check=False)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
        if process.stdin:
            with contextlib.suppress(OSError):
                process.stdin.close()
        self.reader.join(timeout=1)
        if process.stdout:
            process.stdout.close()
        self.log.close()


def _interval(result, cue):
    from .forced_align import _canonical_lexical_surfaces, _alignment_model_text, _lexeme
    words = result.get('word_segments', [])
    expected = [_lexeme(x) for x in _canonical_lexical_surfaces(_alignment_model_text(cue['text']))]
    actual = [_lexeme(str(w.get('word', ''))) for w in words]
    if not words or actual != expected:
        raise ValueError('incomplete_ctc_words')
    previous = -1
    for word in words:
        start = round(float(word['start']) * 1000)
        end = round(float(word['end']) * 1000)
        score = float(word.get('score', -1))
        if not (0.30 <= score <= 1 and 0 < end - start <= 2500
                and start >= previous and cue['start_ms'] - 500 <= start
                and end <= cue['end_ms'] + 500):
            raise ValueError('unusable_ctc_timing')
        previous = end
    start = max(0, round(float(words[0]['start']) * 1000))
    end = round(float(words[-1]['end']) * 1000)
    if end <= start:
        raise ValueError('empty_ctc_interval')
    return start, end


def _worker(audio_path):
    def emit(value):
        print('MAS_ALIGN ' + json.dumps(value, ensure_ascii=False), flush=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import whisperx
            from .forced_align import DEFAULT_TURKISH_ALIGNMENT_MODEL, _alignment_model_text
            model, metadata = whisperx.load_align_model(
                language_code='tr', device='cuda', model_name=DEFAULT_TURKISH_ALIGNMENT_MODEL)
            audio = whisperx.load_audio(audio_path)
        emit({'event': 'ready'})
    except Exception as exc:
        emit({'event': 'unavailable', 'reason': type(exc).__name__})
        return
    for line in sys.stdin:
        cue = json.loads(line)
        answer = {'uid': cue['uid'], 'status': 'fallback', 'reason': 'ctc_rejected'}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = whisperx.align(
                    [{'start': cue['start_ms'] / 1000, 'end': cue['end_ms'] / 1000,
                      'text': _alignment_model_text(cue['text'])}],
                    model, metadata, audio, 'cuda', interpolate_method='ignore',
                    return_char_alignments=False)
                start, end = _interval(result, cue)
            answer.update(status='aligned', start_ms=start, end_ms=end)
        except Exception as exc:
            answer['reason'] = type(exc).__name__
        emit(answer)


def refine_cues(cues, audio_path, journal_root, binding, *, remaining,
                group_seconds=20, total_seconds=600, reserve_seconds=1800, budget_path=None,
                session_factory=None):
    from ..delivery_first import read_signed, write_signed
    from ..reliability import digest
    from ..progress import mark_work_progress
    session_factory = session_factory or BoundedAligner
    root = Path(journal_root)
    root.mkdir(parents=True, exist_ok=True)
    budget_path = Path(budget_path) if budget_path is not None else root / 'budget.json'
    # Each group is charged before launch. A killed parent never grants a free retry.
    budget = (read_signed(budget_path, 'alignment-budget') if budget_path.exists()
              else {'episode': binding['episode'], 'spent_seconds': 0.0, 'limit_seconds': total_seconds})
    if (budget.get('episode') != binding['episode'] or budget.get('limit_seconds') != total_seconds
            or type(budget.get('spent_seconds')) not in (int, float)
            or not math.isfinite(budget['spent_seconds']) or not 0 <= budget['spent_seconds'] <= total_seconds):
        raise ValueError('delivery alignment budget identity changed')
    session = None
    restarts = 0
    refined, warnings = [], []
    try:
        for cue in cues:
            uid = digest({'binding': binding, 'cue': cue})
            path = root / (uid + '.json')
            item = dict(cue)
            if path.exists():
                saved = read_signed(path, 'alignment-group')
                if saved.get('binding') != binding or saved.get('cue') != cue:
                    raise ValueError('delivery alignment checkpoint identity changed')
                result = saved.get('result') or {'status': 'fallback', 'reason': 'interrupted_group'}
            else:
                result = {'status': 'fallback', 'reason': 'delivery_time_reserved'}
                allowance = min(group_seconds, total_seconds - budget['spent_seconds'],
                                max(0, remaining() - reserve_seconds))
                if allowance >= 0.1 and restarts < 2:
                    if session is None:
                        startup = min(90, total_seconds - budget['spent_seconds'],
                                      max(0, remaining() - reserve_seconds))
                        budget['spent_seconds'] += startup
                        write_signed(budget_path, budget, 'alignment-budget')
                        started = time.monotonic()
                        try:
                            session = session_factory(audio_path, root / 'worker.log', startup_seconds=startup)
                        except AlignmentUnavailable as exc:
                            result['reason'] = str(exc)
                            restarts += 1
                        finally:
                            budget['spent_seconds'] -= max(0, startup - (time.monotonic() - started))
                            write_signed(budget_path, budget, 'alignment-budget')
                    allowance = min(group_seconds, total_seconds - budget['spent_seconds'],
                                    max(0, remaining() - reserve_seconds))
                    if session is not None and allowance >= 0.1:
                        budget['spent_seconds'] += allowance
                        write_signed(budget_path, budget, 'alignment-budget')
                        write_signed(path, {'binding': binding, 'cue': cue, 'result': None}, 'alignment-group')
                        started = time.monotonic()
                        try:
                            result = session.align(cue, allowance)
                        except AlignmentUnavailable as exc:
                            result = {'status': 'fallback', 'reason': str(exc)}
                            session.close()
                            session = None
                            restarts += 1
                        finally:
                            budget['spent_seconds'] -= max(0, allowance - (time.monotonic() - started))
                            write_signed(budget_path, budget, 'alignment-budget')
                write_signed(path, {'binding': binding, 'cue': cue, 'result': result}, 'alignment-group')
            if result.get('status') == 'aligned':
                start, end = result.get('start_ms'), result.get('end_ms')
                if (type(start) is not int or type(end) is not int
                        or not max(0, cue['start_ms'] - 500) <= start < end <= cue['end_ms'] + 500):
                    result = {'status': 'fallback', 'reason': 'invalid_ctc_interval'}
                else:
                    item.update(start_ms=start, end_ms=end, timing_source='ctc_cue_bounds',
                                source_interval_ms=[cue['start_ms'], cue['end_ms']])
            if result.get('status') != 'aligned':
                item['timing_source'] = 'source_interval_fallback'
                warnings.append({'uid': cue['uid'], 'start_ms': cue['start_ms'], 'end_ms': cue['end_ms'],
                                 'reason': result.get('reason', 'alignment_unavailable'), 'action': 'kept_source_interval'})
            refined.append(item)
            mark_work_progress('delivery_alignment', completed=len(refined))
    finally:
        if session is not None:
            session.close()
    return refined, warnings


if __name__ == '__main__':
    _worker(sys.argv[1])
