import json
from .config import episode_dir
from .state import load,save
from .hashing import sha256_file,sha256_json
from .packs.build import build_pack
from .packs.validate import validate_returned
from .subtitle.srt import render
from .subtitle.validation import qc
def dirs(ep):
 d=episode_dir(ep)
 for x in ('source','work','translation_input','translation_output','reports','final/subtitles','logs'): (d/x).mkdir(parents=True,exist_ok=True)
 return d
def fixture_records(d):
 p=d/'work/records.json'
 if p.exists(): return json.loads(p.read_text())
 r=[{'block_uid':'fixture-0001','block_index':1,'start_ms':1000,'end_ms':2600,'tr_text':'Merhaba Defne.','id_text':'Halo Defne.'},{'block_uid':'fixture-0002','block_index':2,'start_ms':3000,'end_ms':4800,'tr_text':'Nasılsın?','id_text':'Apa kabar?'}]; p.write_text(json.dumps(r,ensure_ascii=False,indent=2)); return r
def run(ep,source_url=None,fixture=False):
 d=dirs(ep); sp=d/'work/state.json'; state=load(sp,ep)
 if source_url: (d/'source/source.url').write_text(source_url+'\n')
 if fixture: (d/'source/FIXTURE').write_text('offline fixture\n')
 if not (d/'source/FIXTURE').exists() and not (d/'work/records.json').exists(): raise RuntimeError('GPU source acquisition/ASR CLI wiring remains the v0.1.0 RunPod integration blocker')
 records=fixture_records(d); state['stages']['source']={'status':'complete','record_sha256':sha256_json(records)}; save(sp,state)
 trp=d/f'translation_input/Muhtemel Ask {ep}.Bolum_TR_CORRECTION_PACK.zip'; trm=build_pack(trp,ep,'TR',records); state['stages']['tr_pack']={'status':'complete','sha256':sha256_file(trp)}; save(sp,state); trret=d/f'translation_output/Muhtemel Ask {ep}.Bolum_TR_TEXT_CORRECTED.zip'
 if not trret.exists(): print('[WAIT] TR CORRECTION\nGPU WORK COMPLETE\nSafe to stop RunPod now.\n\nWaiting for:\n'+trret.name); return 20
 corrected=validate_returned(trret,trm,records); state['stages']['tr_return']={'status':'complete','sha256':sha256_file(trret)}; save(sp,state)
 idp=d/f'translation_input/Muhtemel Ask {ep}.Bolum_ID_TRANSLATION_PACK.zip'; idm=build_pack(idp,ep,'ID',corrected); state['stages']['id_pack']={'status':'complete','sha256':sha256_file(idp)}; save(sp,state); idret=d/f'translation_output/Muhtemel Ask {ep}.Bolum_ID_TRANSLATED.zip'
 if not idret.exists(): print('[WAIT] ID TRANSLATION\nWaiting for:\n'+idret.name); return 21
 translated=validate_returned(idret,idm,corrected); (d/f'final/subtitles/Muhtemel Ask {ep}.Bolum-tr.srt').write_text(render(translated,'tr_text')); (d/f'final/subtitles/Muhtemel Ask {ep}.Bolum-id.srt').write_text(render(translated,'id_text')); issues=qc(translated); report={'status':'PASS' if not issues else 'PASS_WITH_REVIEW','episode':ep,'final_subtitle_block_count':len(translated),'issues':issues,'pipeline_version':state['pipeline_version'],'schema_version':state['schema_version'],'rules_version':state['rules_version']}; (d/'reports/qc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); (d/'reports/qc.md').write_text(f"# QC\n\nStatus: {report['status']}\n\nIssues: {len(issues)}\n"); state['stages']['qc']={'status':'complete','status_result':report['status']}; save(sp,state); print(f"[9/9] QC {report['status']}"); return 0
def status(ep): print(json.dumps(load(episode_dir(ep)/'work/state.json',ep),indent=2)); return 0
