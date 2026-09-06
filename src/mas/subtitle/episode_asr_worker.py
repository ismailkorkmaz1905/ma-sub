import importlib.metadata
import json
import math
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from ..engine.primary_checkpoint import model_identity, producer_identity
from ..reliability import IntegrityError, atomic_json, digest, file_digest


OPTIONS = {"language": "tr", "word_timestamps": True, "vad_filter": False,
           "condition_on_previous_text": False, "beam_size": 5, "temperature": 0.0}


def consume(request, segments, binding, *, clock=time.monotonic):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    started = clock()
    completed = []
    for index, segment in enumerate(segments):
        if clock() - started >= request['maximum_seconds']:
            raise TimeoutError('full ASR worker deadline expired')
        checkpoint = output / f'segment-{index:06d}.json'
        if checkpoint.exists():
            raise IntegrityError('raw segment exists; preserve prior run instead of overwriting')
        body = {'format': 'mas-episode-draft-asr-segment-1', 'binding_sha256': digest(binding),
                'index': index, 'segment': segment, 'status': 'REVIEW_REQUIRED'}
        atomic_json(checkpoint, {'data': body, 'sha256': digest(body)})
        completed.append({'path': checkpoint.name, 'sha256': file_digest(checkpoint)})
        atomic_json(output / 'progress.json', {'completed_units': index + 3,
                                             'artifact_bytes': 0})
        if index % 10 == 0:
            print(f"ASR segment {index + 1}; source time {segment['end']:.2f} seconds", flush=True)
    return completed, clock() - started


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    episode = request.get('episode')
    maximum = request.get('maximum_seconds')
    if (type(episode) is not int or episode < 1 or isinstance(maximum, bool)
            or not isinstance(maximum, (int, float)) or not math.isfinite(maximum)
            or not 0 < maximum <= 4800):
        raise IntegrityError('invalid Episode draft worker request')
    source, output = Path(request['audio']), Path(request['output'])
    if output.exists():
        raise IntegrityError('ASR output directory already exists')
    output.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    if file_digest(source) != request['audio_sha256']:
        raise IntegrityError('Episode audio differs from source preparation')
    with wave.open(str(source), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise IntegrityError('Episode ASR needs the attested mono16k PCM16 WAV')
        duration = wav.getnframes() / wav.getframerate()
    atomic_json(output / 'progress.json', {'completed_units': 1, 'artifact_bytes': 0})
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from faster_whisper import WhisperModel
    if not torch.cuda.is_available():
        raise RuntimeError('Episode ASR requires CUDA; CPU fallback is forbidden')
    versions = {name: importlib.metadata.version(name).split('+')[0]
                for name in ('torch', 'faster-whisper', 'ctranslate2')}
    if versions != {'torch': '2.8.0', 'faster-whisper': '1.2.1', 'ctranslate2': '4.8.1'}:
        raise IntegrityError('Episode GPU runtime differs from qualified versions')
    identity = model_identity(request['model_dir'])
    if identity['sha256'] != request['model_sha256']:
        raise IntegrityError('Episode ASR model differs from qualified weights')
    binding = {'episode': episode, 'audio_sha256': request['audio_sha256'],
               'duration_seconds': duration, 'model': identity, 'versions': versions,
               'device': 'cuda', 'compute_type': 'float16', 'options': OPTIONS,
               'producer': producer_identity([run_worker, consume]),
               'worker_sha256': file_digest(Path(__file__))}
    atomic_json(output / 'binding.json', {'data': binding, 'sha256': digest(binding)})
    print('Source, model and CUDA runtime verified; loading large-v3', flush=True)
    loaded = time.monotonic()
    model = WhisperModel(request['model_dir'], device='cuda', compute_type='float16',
                         local_files_only=True)
    model_load_seconds = time.monotonic() - loaded
    atomic_json(output / 'progress.json', {'completed_units': 2, 'artifact_bytes': 0})
    try:
        iterator, info = model.transcribe(str(source), **OPTIONS)

        def records():
            for segment in iterator:
                yield {'id': segment.id, 'start': segment.start, 'end': segment.end,
                       'text': segment.text, 'avg_logprob': segment.avg_logprob,
                       'no_speech_prob': segment.no_speech_prob,
                       'words': [{'word': w.word, 'start': w.start, 'end': w.end,
                                  'probability': w.probability} for w in segment.words or []]}

        completed, inference_seconds = consume(request, records(), binding)
        if file_digest(source) != request['audio_sha256'] or model_identity(request['model_dir']) != identity:
            raise IntegrityError('source or model changed during Episode ASR')
        result = {'format': 'mas-episode-draft-asr-result-1', 'episode': episode,
                  'status': 'REVIEW_REQUIRED', 'strict_delivery': False, 'complete': True,
                  'binding_sha256': digest(binding), 'segments': completed,
                  'language': info.language, 'started_at_utc': started_at,
                  'elapsed_seconds': time.monotonic() - started,
                  'model_load_seconds': model_load_seconds, 'inference_seconds': inference_seconds}
        atomic_json(output / 'result.json', {'data': result, 'sha256': digest(result)})
        print(f"Episode {episode} ASR complete: {len(completed)} segments in {inference_seconds:.2f} seconds", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run_worker(sys.argv[1])
