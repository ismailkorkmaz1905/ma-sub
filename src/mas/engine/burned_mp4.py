import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .download import atomic_write_json, sha256_file
from .srt import parse_srt


SUBTITLE_STYLE = 'FontName=Arial,FontSize=14,Outline=0.7,Shadow=0,MarginV=14,MarginL=26,MarginR=26'
ENCODERS = {
    'h264_nvenc': ['-preset', 'p4', '-rc', 'vbr', '-cq', '19', '-b:v', '0'],
    'h264_qsv': ['-preset', 'veryfast', '-global_quality', '18'],
    'libx264': ['-preset', 'fast', '-crf', '18'],
}


def _probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def burn_indonesian_mp4(source_video, id_srt, output_path, *, encoder='h264_nvenc',
                        timeout_seconds=7200):
    source, subtitles, output = map(Path, (source_video, id_srt, output_path))
    if encoder not in ENCODERS or not 0 < timeout_seconds <= 14400:
        raise ValueError('Unsupported MP4 encoder or unbounded timeout')
    if output.suffix.lower() != '.mp4' or output.resolve() in {source.resolve(), subtitles.resolve()}:
        raise ValueError('Burned MP4 requires a separate .mp4 output')
    entries = parse_srt(subtitles)
    if not entries:
        raise ValueError('Burned MP4 requires nonempty validated subtitles')
    inputs = {'source_sha256': sha256_file(source), 'id_srt_sha256': sha256_file(subtitles)}
    receipt_path = output.with_suffix('.burn.json')
    if output.exists():
        if not receipt_path.is_file():
            raise ValueError('Existing MP4 has no bound receipt; preserve it')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('inputs') != inputs or receipt.get('style') != SUBTITLE_STYLE
                or receipt.get('encoder') != encoder or receipt.get('output_sha256') != sha256_file(output)
                or receipt.get('output_bytes') != output.stat().st_size):
            raise ValueError('Existing MP4 differs from requested bound inputs; preserve it')
        return receipt
    before = _probe(source)
    video = next(s for s in before['streams'] if s['codec_type'] == 'video')
    duration = float(before['format']['duration'])
    if any(e.end_ms > round(duration * 1000) + 1 for e in entries):
        raise ValueError('Subtitle timing exceeds immutable source duration')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.burn-', dir=output.parent) as folder:
        work = Path(folder)
        shutil.copyfile(subtitles, work / 'id.srt')
        partial = work / 'encoded.mp4'
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', '-i', str(source.resolve()),
                   '-map', '0:v:0', '-map', '0:a:0', '-sn', '-dn',
                   '-vf', "subtitles=id.srt:force_style='" + SUBTITLE_STYLE + "'",
                   '-c:v', encoder, *ENCODERS[encoder], '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-metadata:s:a:0', 'language=tur',
                   '-movflags', '+faststart', str(partial.resolve())]
        log_path = output.with_suffix('.encode.log')
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=work, stdout=log, stderr=log,
                                       timeout=timeout_seconds, check=False)
        if completed.returncode:
            raise RuntimeError(f'MP4 encoding failed; retained log: {log_path}')
        after = _probe(partial)
        streams = after['streams']
        encoded = next(s for s in streams if s['codec_type'] == 'video')
        audio = next(s for s in streams if s['codec_type'] == 'audio')
        if (len(streams) != 2 or encoded['codec_name'] != 'h264'
                or encoded.get('pix_fmt') != 'yuv420p' or audio['codec_name'] != 'aac'
                or (encoded['width'], encoded['height']) != (video['width'], video['height'])
                or abs(float(after['format']['duration']) - duration) > 0.1):
            raise ValueError('Encoded MP4 stream, dimensions or duration verification failed')
        if sha256_file(source) != inputs['source_sha256'] or sha256_file(subtitles) != inputs['id_srt_sha256']:
            raise ValueError('Source or subtitle changed while encoding')
        receipt = {'format': 'mas-burned-id-mp4-1', 'status': 'VERIFIED_ENCODING',
                   'inputs': inputs, 'encoder': encoder, 'style': SUBTITLE_STYLE,
                   'subtitle_blocks': len(entries), 'output_sha256': sha256_file(partial),
                   'output_bytes': partial.stat().st_size, 'duration_seconds': float(after['format']['duration']),
                   'width': encoded['width'], 'height': encoded['height'],
                   'audio_codec': 'aac', 'audio_language': 'tur', 'subtitle_language': 'id',
                   'subtitles_burned_in': True, 'perceptual_acceptance': 'NOT_ASSERTED',
                   'command': command}
        os.replace(partial, output)
        atomic_write_json(receipt_path, receipt)
    return receipt
