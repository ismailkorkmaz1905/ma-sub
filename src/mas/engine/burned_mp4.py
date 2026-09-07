import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

from .download import atomic_write_json, sha256_file
from .srt import format_timestamp, parse_srt
from ..progress import mark_work_progress

SUBTITLE_STYLE = 'FontName=Arial,FontSize=14,Outline=0.7,Shadow=0,MarginV=14,MarginL=26,MarginR=26'
ENCODERS = {
    'h264_nvenc': ['-preset', 'p4', '-rc', 'vbr', '-cq', '19'],
    'h264_qsv': ['-preset', 'veryfast', '-global_quality', '18'],
    'libx264': ['-preset', 'fast', '-crf', '18'],
}
AUDIO_BITRATE = 192_000
ENCODER_OPTION_KEYS = {
    'h264_nvenc': {'-preset', '-rc', '-cq', '-multipass', '-spatial-aq', '-aq-strength',
                   '-temporal-aq', '-rc-lookahead', '-b_ref_mode'},
    'h264_qsv': {'-preset', '-global_quality', '-look_ahead', '-look_ahead_depth'},
    'libx264': {'-preset', '-crf', '-tune'},
}


def _probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def plan_encoding_settings(source_video, *, encoder='h264_nvenc', target_size_gb=3.0,
                           encoder_options=None):
    if encoder not in ENCODERS:
        raise ValueError('Unsupported MP4 encoder')
    if type(target_size_gb) not in (int, float) or not 0 < target_size_gb <= 100:
        raise ValueError('MP4 soft target must be finite and between 0 and 100 decimal GB')
    options = list(encoder_options) if encoder_options is not None else list(ENCODERS[encoder])
    if not options or len(options) % 2 or not all(isinstance(x, str) and x for x in options):
        raise ValueError('Encoder options must be nonempty option/value pairs')
    for key, value in zip(options[::2], options[1::2]):
        if key not in ENCODER_OPTION_KEYS[encoder] or value.startswith('-'):
            raise ValueError('Encoder option is outside the supported option/value allowlist')
    source = Path(source_video)
    probe = _probe(source)
    try:
        duration = float(probe['format']['duration'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Source duration is invalid') from exc
    if not 0 < duration < float('inf'):
        raise ValueError('Source duration is invalid')
    target_bytes = round(float(target_size_gb) * 1_000_000_000)
    bitrate = max(100_000, round(target_bytes * 8 / duration - AUDIO_BITRATE))
    settings = {'encoder': encoder, 'encoder_options': options, 'target_size_gb': float(target_size_gb),
                'target_bytes': target_bytes, 'target_policy': 'SOFT_TARGET',
                'planned_video_bitrate_bps': bitrate, 'audio_bitrate_bps': AUDIO_BITRATE,
                'source_sha256': sha256_file(source), 'source_duration_seconds': duration}
    settings['identity_sha256'] = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return settings, probe


def create_encoding_samples(source_video, id_srt, output_dir, *, encoder='h264_nvenc',
                            target_size_gb=3.0, encoder_options=None, timeout_seconds=180,
                            idle_timeout_seconds=30):
    if not 0 < timeout_seconds <= 180 or not 0 < idle_timeout_seconds <= timeout_seconds:
        raise ValueError('Sample encoding requires bounded idle and total timeouts')
    source, subtitles, output_dir = Path(source_video), Path(id_srt), Path(output_dir)
    entries = parse_srt(subtitles)
    if len(entries) < 3:
        raise ValueError('Representative encoding samples require at least three subtitle cues')
    settings, probe = plan_encoding_settings(source, encoder=encoder, target_size_gb=target_size_gb,
                                             encoder_options=encoder_options)
    video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    duration = settings['source_duration_seconds']
    sample_duration = min(15.0, duration)
    if sample_duration <= 0:
        raise ValueError('Source duration is invalid')
    anchors = [duration * fraction for fraction in (.1, .5, .9)]
    starts = []
    for anchor in anchors:
        start = max(0.0, min(duration - sample_duration, anchor - sample_duration / 2))
        start_ms, end_ms = round(start * 1000), round((start + sample_duration) * 1000)
        if not any(entry.end_ms > start_ms and entry.start_ms < end_ms for entry in entries):
            nearest = min(entries, key=lambda entry: abs((entry.start_ms + entry.end_ms) / 2000 - anchor))
            start = max(0.0, min(duration - sample_duration,
                                 (nearest.start_ms + nearest.end_ms) / 2000 - sample_duration / 2))
        starts.append(start)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'encoding-samples.json'
    if manifest_path.exists():
        raise ValueError('Encoding sample manifest already exists; preserve it')
    decoder = []
    if encoder == 'h264_nvenc' and video['codec_name'] == 'av1' and video.get('pix_fmt') == 'yuv420p':
        decoder = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-c:v', 'av1_cuvid']
    samples = []
    started_all = time.monotonic()
    encode_elapsed = 0.0
    for ordinal, start in enumerate(starts, 1):
        start_ms, end_ms = round(start * 1000), round((start + sample_duration) * 1000)
        intersecting = [entry for entry in entries if entry.end_ms > start_ms and entry.start_ms < end_ms]
        lines = []
        for index, entry in enumerate(intersecting, 1):
            local_start = max(0, entry.start_ms - start_ms)
            local_end = min(round(sample_duration * 1000), entry.end_ms - start_ms)
            if local_end > local_start:
                lines.extend([str(index), f'{format_timestamp(local_start)} --> {format_timestamp(local_end)}',
                              entry.text, ''])
        sample_srt = output_dir / f'sample-{ordinal}.srt'
        sample_srt.write_text('\n'.join(lines), encoding='utf-8')
        if not lines:
            raise ValueError('Fixed sample interval contains no subtitle cues')
        output = output_dir / f'sample-{ordinal}.mp4'
        log = output_dir / f'sample-{ordinal}.encode.log'
        video_filter = f"subtitles=sample-{ordinal}.srt:force_style='" + SUBTITLE_STYLE + "'"
        if decoder:
            video_filter = 'hwdownload,format=nv12,' + video_filter
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', *decoder, '-ss', f'{start:.3f}',
                   '-i', str(source.resolve()), '-t', f'{sample_duration:.3f}', '-map', '0:v:0',
                   '-map', '0:a:0', '-sn', '-dn', '-vf', video_filter, '-c:v', encoder,
                   *settings['encoder_options'], '-b:v', str(settings['planned_video_bitrate_bps']),
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ac', '2',
                   '-metadata:s:a:0', 'language=tur', '-movflags', '+faststart', '-progress', 'pipe:1',
                   '-stats_period', '1', str(output.resolve())]
        started = time.monotonic()
        remaining = timeout_seconds - (time.monotonic() - started_all)
        if remaining <= 0:
            raise TimeoutError('Shared sample encoding time bound expired')
        _run_encode(command, output_dir, log, output, remaining, min(idle_timeout_seconds, remaining))
        elapsed = time.monotonic() - started
        encode_elapsed += elapsed
        encoded_probe = _probe(output)
        streams = encoded_probe['streams']
        encoded_video = next(stream for stream in streams if stream['codec_type'] == 'video')
        encoded_audio = next(stream for stream in streams if stream['codec_type'] == 'audio')
        if (len(streams) != 2 or encoded_video['codec_name'] != 'h264'
                or encoded_audio['codec_name'] != 'aac'
                or (encoded_video['width'], encoded_video['height']) != (video['width'], video['height'])
                or abs(float(encoded_probe['format']['duration']) - sample_duration) > .1):
            raise ValueError('Encoded sample stream, dimensions or duration verification failed')
        samples.append({'ordinal': ordinal, 'source_start_seconds': start,
                        'source_end_seconds': start + sample_duration,
                        'planned_anchor_seconds': anchors[ordinal - 1],
                        'subtitle_file': sample_srt.name, 'output_file': output.name,
                        'subtitle_sha256': sha256_file(sample_srt), 'subtitle_bytes': sample_srt.stat().st_size,
                        'output_sha256': sha256_file(output), 'output_bytes': output.stat().st_size,
                        'encoded_duration_seconds': float(encoded_probe['format']['duration']),
                        'elapsed_seconds': elapsed})
    elapsed_all = time.monotonic() - started_all
    manifest = {'format': 'mas-encoding-samples-1', 'status': 'REVIEW_REQUIRED',
                'perceptual_acceptance': 'NOT_ASSERTED',
                'inputs': {'source_sha256': settings['source_sha256'],
                           'id_srt_sha256': sha256_file(subtitles)},
                'encoding_settings': settings, 'style': SUBTITLE_STYLE, 'samples': samples,
                'elapsed_seconds': elapsed_all, 'measured_encode_seconds': encode_elapsed,
                'projected_full_output_bytes': round(sum(sample['output_bytes'] for sample in samples)
                                                     * duration / (3 * sample_duration)),
                'projected_full_encode_seconds': encode_elapsed * duration / (3 * sample_duration)}
    if sha256_file(source) != settings['source_sha256'] or sha256_file(subtitles) != manifest['inputs']['id_srt_sha256']:
        raise ValueError('Source or subtitle changed while encoding samples')
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _encoding_hardware():
    ffmpeg = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=10,
                            text=True, encoding='utf-8', errors='replace').stdout.splitlines()[0]
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                         capture_output=True, check=True, timeout=10, text=True,
                         encoding='utf-8', errors='replace').stdout.strip().splitlines()
    if len(gpu) != 1 or not gpu[0]:
        raise RuntimeError('Technical encoder qualification requires exactly one visible GPU')
    return {'ffmpeg_version': ffmpeg, 'gpu': gpu[0]}


def qualify_encoding(source_video, output_dir, *, encoder='h264_nvenc', target_size_gb=3.0,
                     encoder_options=None):
    source, output_dir = Path(source_video), Path(output_dir)
    if encoder != 'h264_nvenc':
        raise ValueError('Technical GPU encoder qualification requires h264_nvenc')
    settings, probe = plan_encoding_settings(source, encoder=encoder, target_size_gb=target_size_gb,
                                             encoder_options=encoder_options)
    duration = settings['source_duration_seconds']
    sample_duration = min(15.0, duration)
    anchors = [duration * fraction for fraction in (.1, .5, .9)]
    technical_srt = output_dir / 'technical-encoder-test.srt'
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, anchor in enumerate(anchors, 1):
        start = max(0, round((anchor - sample_duration / 4) * 1000))
        end = min(round(duration * 1000), start + max(500, round(sample_duration / 2 * 1000)))
        lines.extend([str(index), f'{format_timestamp(start)} --> {format_timestamp(end)}',
                      'TECHNICAL ENCODER TEST - SYNTHETIC SUBTITLE', ''])
    technical_bytes = '\n'.join(lines).encode('utf-8')
    hardware = _encoding_hardware()
    receipt_path = output_dir / 'technical-qualification.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        manifest_path = output_dir / 'encoding-samples.json'
        manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
        samples_valid = all((output_dir / item['output_file']).is_file()
                            and sha256_file(output_dir / item['output_file']) == item['output_sha256']
                            and (output_dir / item['output_file']).stat().st_size == item['output_bytes']
                            and (output_dir / item['subtitle_file']).is_file()
                            and sha256_file(output_dir / item['subtitle_file']) == item['subtitle_sha256']
                            and (output_dir / item['subtitle_file']).stat().st_size == item['subtitle_bytes']
                            for item in receipt.get('samples', []))
        if (receipt.get('settings_identity_sha256') != settings['identity_sha256']
                or receipt.get('source_sha256') != settings['source_sha256']
                or receipt.get('hardware') != hardware or receipt.get('sample_manifest_sha256') != manifest_sha
                or not technical_srt.is_file()
                or receipt.get('technical_subtitle_sha256') != sha256_file(technical_srt)
                or not samples_valid or len(receipt.get('samples', [])) != 3):
            raise ValueError('Existing technical encoder qualification identity changed; preserve it')
        return receipt_path
    if any(output_dir.iterdir()):
        raise ValueError('Technical encoder qualification directory is nonempty; preserve it')
    technical_srt.write_bytes(technical_bytes)
    manifest_path = create_encoding_samples(source, technical_srt, output_dir, encoder=encoder,
                                            target_size_gb=target_size_gb,
                                            encoder_options=encoder_options)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    receipt = {'format': 'mas-technical-encoder-qualification-1', 'status': 'TECHNICALLY_VERIFIED',
               'scope': 'ENCODER_ONLY', 'subtitle_kind': 'SYNTHETIC_TECHNICAL_TEST',
               'perceptual_acceptance': 'NOT_ASSERTED', 'translation_acceptance': 'NOT_ASSERTED',
               'source_sha256': settings['source_sha256'],
               'settings_identity_sha256': settings['identity_sha256'], 'hardware': hardware,
               'technical_subtitle_sha256': sha256_file(technical_srt),
               'sample_manifest_sha256': sha256_file(manifest_path),
               'measured_encode_seconds': manifest['measured_encode_seconds'],
               'projected_full_encode_seconds': manifest['projected_full_encode_seconds'],
               'samples': manifest['samples']}
    atomic_write_json(receipt_path, receipt)
    return receipt_path


def _tree_bytes(root, timeout_seconds=180):
    total = 0
    seen = set()
    deadline = time.monotonic() + timeout_seconds
    pending = [os.fspath(root)]
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError('Network volume usage inspection exceeded its time bound')
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if time.monotonic() > deadline:
                    raise TimeoutError('Network volume usage inspection exceeded its time bound')
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                    continue
                if entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    if not stat.st_ino:
                        stat = os.stat(entry.path, follow_symlinks=False)
                    identity = (stat.st_dev, stat.st_ino)
                    if identity not in seen:
                        seen.add(identity)
                        total += stat.st_size
    return total


def inspect_encoding_storage(path, *, network_volume_root=None, network_volume_quota_bytes=None,
                             usage_timeout_seconds=180):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    result = {'path': str(path.resolve()), 'filesystem_capacity_bytes': usage.total,
              'filesystem_used_bytes': usage.used, 'filesystem_free_bytes': usage.free}
    if network_volume_root is not None:
        root = Path(network_volume_root).resolve()
        if not root.is_dir():
            raise ValueError('Declared network volume root does not exist')
        if not path.resolve().is_relative_to(root):
            raise ValueError('Storage path is outside the declared network volume')
        if not isinstance(network_volume_quota_bytes, int) or network_volume_quota_bytes <= 0:
            raise ValueError('Declared network volume requires its actual quota in bytes')
        if not 0 < usage_timeout_seconds <= 300:
            raise ValueError('Network volume usage inspection requires a bounded timeout')
        scan_started = time.monotonic()
        actual = _tree_bytes(root, usage_timeout_seconds)
        result.update({'network_volume_root': str(root), 'network_volume_quota_bytes': network_volume_quota_bytes,
                       'network_volume_used_bytes': actual,
                       'network_volume_free_bytes': max(0, network_volume_quota_bytes - actual),
                       'network_volume_scan_elapsed_seconds': round(time.monotonic() - scan_started, 3)})
    return result


def _run_encode(command, cwd, log_path, partial, total_timeout, idle_timeout):
    updates = queue.Queue()

    def read(stream):
        for line in stream:
            updates.put(line.rstrip())
        updates.put(None)
    process = thread = None
    failure = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=log, text=True, encoding='utf-8', errors='replace')
            thread = threading.Thread(target=read, args=(process.stdout,), daemon=True)
            thread.start()
            started = last_progress = time.monotonic()
            frame = out_time = -1
            while process.poll() is None:
                now = time.monotonic()
                if now - started > total_timeout:
                    failure = 'total time bound expired'
                    break
                if now - last_progress > idle_timeout:
                    failure = 'frame progress watchdog expired'
                    break
                try:
                    line = updates.get(timeout=.25)
                except queue.Empty:
                    continue
                if line is None:
                    continue
                log.write('[progress] ' + line + '\n')
                key, _, value = line.partition('=')
                try:
                    number = int(value)
                    advanced = ((key == 'frame' and number > frame)
                                or (key in ('out_time_us', 'out_time_ms') and number > out_time))
                    if key == 'frame' and number > frame:
                        frame = number
                    elif key in ('out_time_us', 'out_time_ms') and number > out_time:
                        out_time = number
                    if advanced:
                        last_progress = time.monotonic()
                        mark_work_progress('burned_mp4', completed=max(frame, 0))
                except ValueError:
                    pass
            if failure:
                process.kill()
            returncode = process.wait(timeout=10)
            if failure:
                log.write('[watchdog] ' + failure + '\n')
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        _preserve_failed_partial(partial, log_path)
        raise
    finally:
        if process is not None and process.stdout:
            process.stdout.close()
        if thread is not None:
            thread.join(timeout=1)
    if failure or returncode:
        _preserve_failed_partial(partial, log_path)
        raise RuntimeError(f'MP4 encoding failed ({failure or f"ffmpeg exit code {returncode}"}); retained log: {log_path}')


def _preserve_failed_partial(partial, log_path):
    if partial.is_file() and partial.stat().st_size:
        failed = log_path.with_suffix('.failed.mp4')
        try:
            shutil.copyfile(partial, failed)
        except OSError as exc:
            try:
                with log_path.open('a', encoding='utf-8') as log:
                    log.write(f'[diagnostic] partial preservation failed: {exc}\n')
            except OSError:
                pass


def _stage_output(partial, output):
    staged = output.with_suffix('.encode.partial.mp4')
    try:
        if partial.stat().st_dev == output.parent.stat().st_dev:
            os.replace(partial, staged)
        else:
            shutil.copyfile(partial, staged)
        os.replace(staged, output)
    except BaseException:
        if staged.exists():
            failed = output.with_suffix('.encode.interrupted.mp4')
            os.replace(staged, failed)
        raise


def burn_indonesian_mp4(source_video, id_srt, output_path, *, encoder='h264_nvenc',
                        timeout_seconds=7200, target_size_gb=3.0, encoder_options=None,
                        sample_approval_path=None, idle_timeout_seconds=900, scratch_dir=None,
                        network_volume_root=None, network_volume_quota_bytes=None,
                        require_sample_approval=False):
    source, subtitles, output = map(Path, (source_video, id_srt, output_path))
    if not 0 < timeout_seconds <= 14400:
        raise ValueError('Unsupported MP4 encoder or unbounded timeout')
    if not 0 < idle_timeout_seconds <= timeout_seconds:
        raise ValueError('MP4 progress watchdog must be positive and within total timeout')
    settings, before = plan_encoding_settings(source, encoder=encoder, target_size_gb=target_size_gb,
                                              encoder_options=encoder_options)
    options = settings['encoder_options']
    subtitle_sha256 = sha256_file(subtitles)
    approval = None
    if encoder_options is not None or require_sample_approval:
        approval_path = Path(sample_approval_path) if sample_approval_path else None
        if not approval_path or not approval_path.is_file():
            raise ValueError('Final encoder settings require an explicit sample approval record')
        try:
            approval_record = json.loads(approval_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('Sample approval record must be valid UTF-8 JSON') from exc
        if (approval_record.get('approved') is not True
                or approval_record.get('encoding_settings_identity_sha256') != settings['identity_sha256']
                or approval_record.get('source_sha256') != settings['source_sha256']
                or approval_record.get('source_duration_seconds') != settings['source_duration_seconds']
                or approval_record.get('id_srt_sha256') != subtitle_sha256
                or approval_record.get('style') != SUBTITLE_STYLE):
            raise ValueError('Sample approval record does not approve this source and encoder plan')
        manifest_relative = approval_record.get('sample_manifest_path')
        if (not isinstance(manifest_relative, str)
                or not manifest_relative.startswith('work/encoding-samples/')
                or '\\' in manifest_relative or '..' in Path(manifest_relative).parts):
            raise ValueError('Sample approval manifest path is outside episode work samples')
        episode_root = approval_path.parent.parent.resolve()
        manifest_path = (episode_root / manifest_relative).resolve()
        if not manifest_path.is_relative_to(episode_root / 'work' / 'encoding-samples'):
            raise ValueError('Sample approval manifest path is outside episode work samples')
        if (not manifest_path.is_file()
                or approval_record.get('sample_manifest_sha256') != sha256_file(manifest_path)):
            raise ValueError('Sample approval manifest identity changed')
        sample_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (sample_manifest.get('encoding_settings') != settings
                or sample_manifest.get('style') != SUBTITLE_STYLE
                or sample_manifest.get('inputs') != {'source_sha256': settings['source_sha256'],
                                                     'id_srt_sha256': subtitle_sha256}):
            raise ValueError('Sample manifest does not match final encoder inputs')
        for sample in sample_manifest.get('samples', []):
            output_name, subtitle_name = sample.get('output_file'), sample.get('subtitle_file')
            if (not isinstance(output_name, str) or Path(output_name).name != output_name
                    or not isinstance(subtitle_name, str) or Path(subtitle_name).name != subtitle_name):
                raise ValueError('Sample manifest artifact path is invalid')
            sample_path = manifest_path.parent / output_name
            subtitle_path = manifest_path.parent / subtitle_name
            if (not sample_path.is_file() or sample_path.stat().st_size != sample.get('output_bytes')
                    or sha256_file(sample_path) != sample.get('output_sha256')
                    or not subtitle_path.is_file() or subtitle_path.stat().st_size != sample.get('subtitle_bytes')
                    or sha256_file(subtitle_path) != sample.get('subtitle_sha256')):
                raise ValueError('Approved encoder sample bytes changed')
        if len(sample_manifest.get('samples', [])) != 3:
            raise ValueError('Sample manifest must bind exactly three encoder samples')
        approval = {'path': str(approval_path.resolve()), 'sha256': sha256_file(approval_path),
                    'bytes': approval_path.stat().st_size}
    if output.suffix.lower() != '.mp4' or output.resolve() in {source.resolve(), subtitles.resolve()}:
        raise ValueError('Burned MP4 requires a separate .mp4 output')
    entries = parse_srt(subtitles)
    if not entries:
        raise ValueError('Burned MP4 requires nonempty validated subtitles')
    video = next(s for s in before['streams'] if s['codec_type'] == 'video')
    duration = settings['source_duration_seconds']
    if any(e.end_ms > round(duration * 1000) + 1 for e in entries):
        raise ValueError('Subtitle timing exceeds immutable source duration')
    target_bytes = settings['target_bytes']
    bitrate = settings['planned_video_bitrate_bps']
    inputs = {'source_sha256': settings['source_sha256'], 'id_srt_sha256': subtitle_sha256}
    receipt_path = output.with_suffix('.burn.json')
    if output.exists():
        if not receipt_path.is_file():
            raise ValueError('Existing MP4 has no bound receipt; preserve it')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('inputs') != inputs or receipt.get('style') != SUBTITLE_STYLE
                or receipt.get('encoding_settings') != settings or receipt.get('sample_approval') != approval
                or receipt.get('output_sha256') != sha256_file(output)
                or receipt.get('output_bytes') != output.stat().st_size):
            raise ValueError('Existing MP4 differs from requested bound inputs or encoding; preserve it')
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch_dir) if scratch_dir else output.parent
    output_storage = inspect_encoding_storage(output.parent, network_volume_root=network_volume_root,
                                              network_volume_quota_bytes=network_volume_quota_bytes)
    scratch_storage = inspect_encoding_storage(scratch)
    required = round(target_bytes * 1.05)
    output_free = output_storage.get('network_volume_free_bytes', output_storage['filesystem_free_bytes'])
    if output_free < required or scratch_storage['filesystem_free_bytes'] < required:
        raise ValueError('Insufficient storage for planned soft-target encode')
    with tempfile.TemporaryDirectory(prefix='.burn-', dir=scratch) as folder:
        work = Path(folder)
        shutil.copyfile(subtitles, work / 'id.srt')
        partial = work / 'encoded.mp4'
        decoder = []
        video_filter = "subtitles=id.srt:force_style='" + SUBTITLE_STYLE + "'"
        if encoder == 'h264_nvenc' and video['codec_name'] == 'av1' and video.get('pix_fmt') == 'yuv420p':
            decoder = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-c:v', 'av1_cuvid']
            video_filter = 'hwdownload,format=nv12,' + video_filter
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', *decoder, '-i', str(source.resolve()),
                   '-map', '0:v:0', '-map', '0:a:0', '-sn', '-dn', '-vf', video_filter,
                   '-c:v', encoder, *options, '-b:v', str(bitrate), '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-metadata:s:a:0', 'language=tur',
                   '-movflags', '+faststart', '-progress', 'pipe:1', '-stats_period', '1', str(partial.resolve())]
        log_path = output.with_suffix('.encode.log')
        encode_started = time.monotonic()
        _run_encode(command, work, log_path, partial, timeout_seconds, idle_timeout_seconds)
        encode_elapsed = time.monotonic() - encode_started
        after = _probe(partial)
        streams = after['streams']
        encoded = next(s for s in streams if s['codec_type'] == 'video')
        audio = next(s for s in streams if s['codec_type'] == 'audio')
        if (len(streams) != 2 or encoded['codec_name'] != 'h264' or encoded.get('pix_fmt') != 'yuv420p'
                or audio['codec_name'] != 'aac' or (encoded['width'], encoded['height']) != (video['width'], video['height'])
                or abs(float(after['format']['duration']) - duration) > .1):
            raise ValueError('Encoded MP4 stream, dimensions or duration verification failed')
        if sha256_file(source) != inputs['source_sha256'] or sha256_file(subtitles) != inputs['id_srt_sha256']:
            raise ValueError('Source or subtitle changed while encoding')
        if approval and (sha256_file(approval['path']) != approval['sha256']
                         or Path(approval['path']).stat().st_size != approval['bytes']):
            raise ValueError('Sample approval record changed while encoding')
        receipt = {'format': 'mas-burned-id-mp4-2', 'status': 'VERIFIED_ENCODING', 'inputs': inputs,
                   'encoder': encoder, 'encoding_settings': settings, 'sample_approval': approval,
                   'style': SUBTITLE_STYLE,
                   'subtitle_blocks': len(entries), 'output_sha256': sha256_file(partial),
                   'output_bytes': partial.stat().st_size, 'duration_seconds': float(after['format']['duration']),
                   'encode_elapsed_seconds': encode_elapsed,
                   'width': encoded['width'], 'height': encoded['height'], 'audio_codec': 'aac',
                   'audio_language': 'tur', 'subtitle_language': 'id', 'subtitles_burned_in': True,
                   'perceptual_acceptance': 'NOT_ASSERTED', 'storage_preflight': {
                       'required_free_bytes': required, 'output': output_storage, 'scratch': scratch_storage},
                   'command': command}
        _stage_output(partial, output)
        atomic_write_json(receipt_path, receipt)
    return receipt
