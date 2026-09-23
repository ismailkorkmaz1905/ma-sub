"""Download -> optional publisher captions / GPU ASR -> ChatGPT -> burned MP4."""
from pathlib import Path
import json
import subprocess
import zipfile

from . import colab_flow as f, official_subtitles as official
from .engine.burned_mp4 import burn_indonesian_mp4


def prepare(root, episode, config, *, source_file='', source_url='', force_asr=False):
    root=Path(root)
    source,saved=f.acquire(root,episode,source_file,source_url)
    route_path=root/'video_workflow.json'
    if route_path.exists():
        route=f.read_json(route_path)
        if route['source_sha256']!=saved['sha256']:raise f.ContractError('Video source changed')
        work=root/route['work_directory']
        if route['route']=='publisher':official.prepare(work,episode,config)
        else:f.prepare(work,episode,config,source_file=str(source))
        return work/'handoff'/f'Muhtemel Ask {episode}.Bolum_TRANSLATION_PACK.zip'
    reason='ASR explicitly requested or source is a different edit'
    # Publisher cues may only accompany the exact publisher asset, never a YouTube cut.
    if not force_asr and saved.get('url','').startswith('https://vmcdn.ciner.com.tr/'):
        try:
            ready=official.prepare(root/'captions',episode,config)
            if ready['status']=='READY':
                binding=f.read_json(root/'captions/source.json')
                probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_format','-of','json',str(source)],
                                                capture_output=True,check=True,timeout=30).stdout)['format']
                if (binding['media_url']!=saved['url'] or abs(float(probe['duration'])*1000-binding['duration_ms'])>1000
                        or abs(float(probe.get('start_time',0)))>.1):
                    raise f.ContractError('Publisher captions do not match this video timeline')
                f.write_json(route_path,dict(route='publisher',work_directory='captions',source_sha256=saved['sha256']))
                return root/'captions/handoff'/f'Muhtemel Ask {episode}.Bolum_TRANSLATION_PACK.zip'
            reason=ready['status']
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            reason=type(exc).__name__+': '+str(exc)
    print('Publisher captions unavailable/unusable; starting GPU ASR:',reason,flush=True)
    # Persist this choice BEFORE inference. Captions arriving later must not replace ASR work.
    f.write_json(route_path,dict(route='asr',work_directory='.',source_sha256=saved['sha256'],reason=reason))
    return f.prepare(root,episode,config,source_file=str(source))


def finish(root, returned_zip, *, encoder='h264_nvenc', target_size_gb=3.0):
    root=Path(root);route=f.read_json(root/'video_workflow.json')
    if f.file_hash(root/'source.mp4')!=route['source_sha256']:raise f.ContractError('Video changed')
    work=root/route['work_directory']
    if route['route']=='publisher':
        schema=f.read_json(work/'schema.json');manifest=f.read_json(work/'manifest.json')
        f.read_return(returned_zip,schema,manifest)
        with zipfile.ZipFile(returned_zip) as archive:
            for batch in manifest['batches']:
                name='translated_'+batch['filename']
                f.atomic_bytes(work/'translations'/name,archive.read(name))
        report=official.finish(work)
        item=next(x for x in report['files'] if '_ID' in x['name'])
        subtitle=work/'output'/item['name']
        draft=True  # Publisher timing preservation alone is not listening verification.
    else:
        report=f.finalize(work,returned_zip)
        item=next(x for x in report['output_files'] if '_ID' in x['path'])
        subtitle=work/item['path'];draft=bool(report['issues'])
    if f.file_hash(subtitle)!=item['sha256']:raise f.ContractError('Subtitle changed')
    episode=f.read_json(root/'source.json')['episode']
    identity=f.digest(dict(source=route['source_sha256'],subtitle=item['sha256'],encoder=encoder,target=target_size_gb))[:16]
    output=root/'output'/identity/(f'Muhtemel_Ask_{episode}_ID'+('.draft' if draft else '')+'.mp4')
    receipt=burn_indonesian_mp4(root/'source.mp4',subtitle,output,encoder=encoder,
                               target_size_gb=target_size_gb,timeout_seconds=14400)
    result=dict(status='BURNED_DRAFT' if draft else 'BURNED_REVIEWED',path=str(output),
                source_path=str(root/'source.mp4'),receipt=receipt,subtitle_report=report,
                delivery='Mounted filesystem verified; independent Drive readback still required')
    f.write_json(root/'video_output.json',result)
    print('Burned MP4:',output,flush=True)
    return result
