import importlib.metadata
import json
import os
import sys
import time
import wave
from pathlib import Path

from ..engine.forced_align import _alignment_model_text
from ..engine.primary_checkpoint import model_identity as asr_identity
from ..reliability import IntegrityError, atomic_json, digest, file_digest
from .episode_alignment_worker import model_identity as ctc_identity


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(request['manifest']).read_text(encoding='utf-8'))
    episode = manifest.get('data', {}).get('episode')
    if (manifest.get('sha256') != digest(manifest.get('data'))
            or manifest['sha256'] != request['manifest_sha256']
            or type(episode) is not int or episode < 1 or not 0 < request['maximum_seconds'] <= 1500):
        raise IntegrityError('invalid Episode acoustic repair request')
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    from whisperx.alignment import align, load_align_model
    if not torch.cuda.is_available():
        raise RuntimeError('acoustic repair requires CUDA')
    versions = {name: importlib.metadata.version(name).split('+')[0]
                for name in ('faster-whisper', 'ctranslate2', 'whisperx', 'torch')}
    if versions != {'faster-whisper': '1.2.1', 'ctranslate2': '4.8.1', 'whisperx': '3.8.6', 'torch': '2.8.0'}:
        raise IntegrityError('acoustic repair runtime differs')
    print('Verifying qualified ASR weights for targeted repair', flush=True)
    asr_model = asr_identity(request['asr_model_dir'])
    if asr_model['sha256'] != request['asr_model_sha256']:
        raise IntegrityError('repair ASR weights changed')
    print('ASR weights verified; verifying pinned CTC weights', flush=True)
    ctc_model = ctc_identity(request['ctc_model_dir'])
    if ctc_model['sha256'] != request['ctc_model_sha256']:
        raise IntegrityError('repair CTC weights changed')
    options = {'language': 'tr', 'word_timestamps': True, 'vad_filter': True,
               'vad_parameters': {'min_silence_duration_ms': 500},
               'condition_on_previous_text': False, 'beam_size': 5, 'temperature': 0.0,
               'hallucination_silence_threshold': 2.0,
               'initial_prompt': 'Defne, Kadir Emindağ, Tolga, Levent Bartıner, Mine, Melis, Özlem, Selim, Selma, Sultan, Zeynep, Leyla, Hamza.'}
    binding = {'manifest_sha256': manifest['sha256'], 'asr_model': asr_model,
               'ctc_model': ctc_model, 'options': options, 'versions': versions,
               'device': 'cuda', 'worker_sha256': file_digest(Path(__file__))}
    atomic_json(output / 'binding.json', {'data': binding, 'sha256': digest(binding)})
    print('Loading ASR and CTC models on CUDA', flush=True)
    decoder = WhisperModel(request['asr_model_dir'], device='cuda', compute_type='float16', local_files_only=True)
    aligner, metadata = load_align_model('tr', 'cuda', model_name=request['ctc_model_dir'], model_cache_only=True)
    completed = []
    try:
        for index, clip in enumerate(manifest['data']['clips']):
            if time.monotonic() - started >= request['maximum_seconds']:
                raise TimeoutError('targeted acoustic repair deadline expired')
            audio_path = Path(request['clips'][clip['name']])
            if file_digest(audio_path) != clip['audio_sha256']:
                raise IntegrityError('repair clip changed after verified transfer')
            with wave.open(str(audio_path), 'rb') as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise IntegrityError('repair clip must be mono PCM16 16 kHz')
                duration = wav.getnframes() / 16000
                if abs(duration - clip['duration_ms'] / 1000) > 1e-9 or not 0 < duration <= 120:
                    raise IntegrityError('repair clip duration differs')
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
            iterator, info = decoder.transcribe(audio, **options)
            segments = [{'id': s.id, 'start': s.start, 'end': s.end, 'text': s.text,
                         'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob,
                         'words': [{'word': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability}
                                   for w in s.words or []]} for s in iterator]
            aligned = []
            for number, segment in enumerate(segments):
                coarse = {'start': max(0, segment['start'] - .7),
                          'end': min(duration, segment['end'] + .7),
                          'text': _alignment_model_text(segment['text'])}
                aligned.append({'index': number, 'input_segment_sha256': digest(segment), 'coarse': coarse,
                                'raw': align([coarse], aligner, metadata, audio, 'cuda', interpolate_method='ignore')})
            if file_digest(audio_path) != clip['audio_sha256']:
                raise IntegrityError('repair clip changed during ASR or alignment')
            body = {'clip': clip, 'binding_sha256': digest(binding), 'segments': segments,
                    'aligned': aligned, 'language': info.language, 'status': 'REVIEW_REQUIRED'}
            path = output / (clip['name'] + '.json')
            atomic_json(path, {'data': body, 'sha256': digest(body)})
            completed.append({'name': clip['name'], 'path': path.name, 'sha256': file_digest(path)})
            print(f'Repaired clip {index + 1}/{len(manifest["data"]["clips"])}: {len(segments)} segments', flush=True)
        print('Repair inference complete; verifying model identities again', flush=True)
        if asr_identity(request['asr_model_dir']) != asr_model or ctc_identity(request['ctc_model_dir']) != ctc_model:
            raise IntegrityError('repair model changed during inference')
        for clip in manifest['data']['clips']:
            if file_digest(Path(request['clips'][clip['name']])) != clip['audio_sha256']:
                raise IntegrityError('repair clip changed before final result')
        body = {'episode': episode, 'complete': True, 'clips': completed, 'segments': completed,
                'elapsed_seconds': time.monotonic() - started, 'binding_sha256': digest(binding),
                'status': 'REVIEW_REQUIRED', 'strict_delivery': False}
        atomic_json(output / 'result.json', {'data': body, 'sha256': digest(body)})
    finally:
        del decoder, aligner
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run_worker(sys.argv[1])
