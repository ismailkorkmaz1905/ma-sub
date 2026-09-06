import importlib.metadata
import json
import os
import sys
import time
import wave
from pathlib import Path

from ..engine.forced_align import _alignment_model_text
from ..reliability import IntegrityError, atomic_json, digest, file_digest


def model_identity(model_dir):
    root = Path(model_dir)
    required = {'config.json', 'preprocessor_config.json', 'pytorch_model.bin',
                'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json'}
    files = {path.relative_to(root).as_posix(): file_digest(path)
             for path in sorted(root.rglob('*')) if path.is_file()}
    if set(files) != required:
        raise IntegrityError('Turkish CTC model file set differs from pinned manifest')
    return {'files': files, 'sha256': digest(files)}


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    episode = request.get('episode')
    maximum = request['maximum_seconds']
    if type(episode) is not int or episode < 1 or not 0 < maximum <= 2400:
        raise IntegrityError('invalid Episode alignment deadline')
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = Path(request['audio'])
    if file_digest(source) != request['audio_sha256']:
        raise IntegrityError('alignment source changed')
    transcript = json.loads(Path(request['transcript']).read_text(encoding='utf-8'))
    if transcript.get('sha256') != digest(transcript.get('data')):
        raise IntegrityError('alignment transcript checksum differs')
    if transcript['data']['audio_sha256'] != request['audio_sha256']:
        raise IntegrityError('alignment transcript audio differs')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from whisperx.alignment import align, load_align_model
    if not torch.cuda.is_available():
        raise RuntimeError('Episode alignment requires CUDA')
    if importlib.metadata.version('whisperx') != '3.8.6':
        raise IntegrityError('alignment WhisperX version differs')
    model_path = Path(request['model_dir'])
    identity = model_identity(model_path)
    if identity['sha256'] != request['model_sha256']:
        raise IntegrityError('Turkish CTC model differs from uploaded pinned model')
    binding = {'episode': episode, 'audio_sha256': request['audio_sha256'],
               'transcript_sha256': transcript['sha256'], 'model': identity,
               'whisperx_version': '3.8.6', 'device': 'cuda', 'interpolate_method': 'ignore',
               'padding_seconds': .7, 'worker_sha256': file_digest(Path(__file__))}
    atomic_json(output / 'binding.json', {'data': binding, 'sha256': digest(binding)})
    with wave.open(str(source), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise IntegrityError('alignment audio must be PCM16 mono 16 kHz')
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
    print('Loading cached Turkish CTC model on CUDA', flush=True)
    model, metadata = load_align_model('tr', 'cuda', model_name=str(model_path), model_cache_only=True)
    completed = []
    try:
        for index, segment in enumerate(transcript['data']['segments']):
            if time.monotonic() - started >= maximum:
                raise TimeoutError('Episode alignment worker deadline expired')
            coarse = {'start': max(0, segment['start'] - .7),
                      'end': min(len(audio) / 16000, segment['end'] + .7),
                      'text': _alignment_model_text(segment['text'])}
            raw = align([coarse], model, metadata, audio, 'cuda', interpolate_method='ignore')
            body = {'format': 'mas-episode-draft-ctc-segment-1', 'index': index,
                    'binding_sha256': digest(binding), 'input_segment_sha256': digest(segment),
                    'coarse': coarse, 'raw': raw, 'status': 'REVIEW_REQUIRED'}
            path = output / f'segment-{index:06d}.json'
            atomic_json(path, {'data': body, 'sha256': digest(body)})
            completed.append({'path': path.name, 'sha256': file_digest(path)})
            if index % 25 == 0:
                print(f'CTC segment {index + 1}; source time {segment["end"]:.2f} seconds', flush=True)
        if model_identity(model_path) != identity or file_digest(source) != request['audio_sha256']:
            raise IntegrityError('alignment model or source changed')
        body = {'episode': episode, 'complete': True, 'segments': completed,
                'binding_sha256': digest(binding), 'elapsed_seconds': time.monotonic() - started,
                'strict_delivery': False, 'status': 'REVIEW_REQUIRED'}
        atomic_json(output / 'result.json', {'data': body, 'sha256': digest(body)})
        print(f'CTC completed: {len(completed)} segments', flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run_worker(sys.argv[1])
