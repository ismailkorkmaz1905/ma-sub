import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import wave
import zipfile
from pathlib import Path

import pytest
from mas import colab_flow as f


def words():
    return [dict(word_id=i,text=t,start_ms=s,end_ms=e,probability=.95,segment_id=seg,risk_flags=[])
            for i,(t,s,e,seg) in enumerate([
                ('Defne,',1000,1400,'a'),('gel.',1500,1800,'a'),
                ('Seni',2700,3000,'b'),('bekliyorum.',3050,3500,'b')])]


def schema():
    return f.build_schema(15,'a'*64,6000,words(),[dict(start_ms=1000,end_ms=1900),dict(start_ms=2700,end_ms=3500)],
        {'forbidden_name_variants':{'Bartıner':['Bartiner']},'religious_terms':{'terms':[]}},'instructions',{'model':'test-only'})


def setup_return(root,s=None,transform=None):
    s=s or schema();pack=f.make_pack(root,s,'instructions',batch_size=1)
    manifest=f.read_json(root/'manifest.json')
    rows=[dict(block_uid=c['block_uid'],schema_sha256=f.digest(s),tr_final=c['primary_text'],
               id_final=t,review_required=False,note='')
          for c,t in zip(s['cues'],['Defne, sini.','Aku menunggumu.'])]
    if transform:transform(rows)
    report=dict(schema_sha256=f.digest(s),total_input_blocks=len(rows),total_output_blocks=len(rows),
                missing_block_count=0,duplicate_block_count=0,review_required_count=sum(r['review_required'] is True for r in rows))
    returned=root/'return.zip'
    with zipfile.ZipFile(returned,'w') as z:
        for b,r in zip(manifest['batches'],rows):z.writestr('translated_'+b['filename'],f.canonical(r)+b'\n')
        z.writestr('translation_report.json',f.canonical(report))
    return s,manifest,rows,returned,pack


def reviewed(s,returned):
    r=f.review_template(s,f.file_hash(returned));r['samples_reviewed']=f.sample_ids(s);r['full_playback_reviewed']=True
    return r


def test_no_early_start_or_premature_end_and_every_word_once():
    ws=words();cues=f.make_cues(ws,6000,'a'*64)
    assert [i for c in cues for i in c['word_ids']]==list(range(len(ws)))
    for c in cues:
        owned=[ws[i] for i in c['word_ids']]
        assert c['start_ms']==owned[0]['start_ms']
        assert max(w['end_ms'] for w in owned)<=c['end_ms']<=max(w['end_ms'] for w in owned)+200
    assert cues[0]['end_ms']<=cues[1]['start_ms']


def test_overlap_never_fixed_by_cutting_speech():
    ws=words();ws[1]['end_ms']=2900
    cues=f.make_cues(ws,6000,'a'*64)
    assert cues[0]['end_ms']==2900
    assert 'overlapping_speech' in cues[0]['risk_flags']


def test_long_pause_breaks_cue_and_zero_word_is_reviewed():
    ws=words();ws[1]['start_ms']=ws[1]['end_ms']=1700
    cues=f.make_cues(ws,6000,'a'*64)
    assert any('zero_duration_word' in c['risk_flags'] for c in cues)
    assert len(cues)>=2


@pytest.mark.parametrize('value',[True,-1,7000,float('nan')])
def test_bad_word_times_rejected(value):
    ws=words();ws[0]['start_ms']=value
    with pytest.raises(f.ContractError):f.make_cues(ws,6000,'a'*64)


def test_word_order_and_probability_are_not_silently_fixed():
    ws=words();ws[1]['start_ms']=900
    with pytest.raises(f.ContractError):f.make_cues(ws,6000,'a'*64)
    ws=words();ws[0]['probability']=float('inf')
    with pytest.raises(f.ContractError):f.make_cues(ws,6000,'a'*64)


def test_uncovered_speech_not_dropped_and_early_start_flagged():
    s=schema();gaps=f.uncovered_speech([dict(start_ms=0,end_ms=10000)],s['words'])
    assert gaps[-1]['end_ms']==10000
    ws=words();ws[0]['start_ms']=0
    s=f.build_schema(15,'a'*64,6000,ws,[dict(start_ms=1000,end_ms=1900)],{},'i',{})
    assert 'early_start_against_vad' in s['cues'][0]['risk_flags']


def test_chunk_cuts_use_silence_and_flag_continuous_speech():
    speech=[dict(start_ms=0,end_ms=299000),dict(start_ms=301000,end_ms=730000)]
    plan=f.chunk_plan(speech,730000)
    assert plan[0]['end_ms']==300000 and not plan[0]['unsafe_end']
    assert plan[1]['unsafe_end']
    assert [(p['start_ms'],p['end_ms']) for p in plan]==[(0,300000),(300000,600000),(600000,730000)]


def test_pack_contains_original_instructions_and_bound_context(tmp_path):
    s,m,rows,returned,pack=setup_return(tmp_path)
    with zipfile.ZipFile(pack) as z:
        assert z.read('TRANSLATION_INSTRUCTIONS.md')==b'instructions'
        for name,h in m['file_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        row=f.strict_json(z.read('batch_001.jsonl'))
        assert row['read_only_context'] and row['verification_text'] is None
    assert f.read_return(returned,s,m)==rows
    assert f.file_hash(f.make_pack(tmp_path,s,'instructions',batch_size=1))==f.file_hash(pack)


@pytest.mark.parametrize('change',[
    lambda r:r[0].update(block_uid='foreign'),
    lambda r:r[0].update(schema_sha256='b'*64),
    lambda r:r[0].update(start_ms=0),
    lambda r:r[0].update(review_required='false'),
    lambda r:r[0].update(id_final=''),
    lambda r:r[0].update(id_final='<script>x</script>'),
    lambda r:r.reverse(),
])
def test_return_mutations_rejected(tmp_path,change):
    s,m,rows,ret,_=setup_return(tmp_path,transform=change)
    with pytest.raises((f.ContractError,TypeError)):f.read_return(ret,s,m)


def test_zip_extra_duplicate_and_traversal_rejected(tmp_path):
    s,m,rows,ret,_=setup_return(tmp_path)
    with zipfile.ZipFile(ret,'a') as z:z.writestr('../attack.json','{}')
    with pytest.raises(f.ContractError):f.read_return(ret,s,m)


def test_json_duplicate_keys_and_nonfinite_rejected():
    for payload in ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}']:
        with pytest.raises(f.ContractError):f.strict_json(payload)


def test_instructions_and_schema_cannot_change_under_return(tmp_path):
    s,_,_,_,_=setup_return(tmp_path)
    with pytest.raises(f.ContractError):f.make_pack(tmp_path,s,'changed')
    s['cues'][0]['end_ms']+=1
    with pytest.raises(f.ContractError):f.make_pack(tmp_path,s,'instructions')


def test_finalize_is_draft_until_listening_and_no_silent_truncation(tmp_path):
    s,_,_,ret,_=setup_return(tmp_path)
    result=f.finalize(tmp_path,ret)
    assert result['status']=='DRAFT_REVIEW_REQUIRED'
    assert all('.draft.srt' in p['path'] for p in result['output_files'])
    f.write_json(tmp_path/'review.json',reviewed(s,ret))
    result=f.finalize(tmp_path,ret)
    assert result['status']=='REVIEWED'
    for p in result['output_files']:
        assert '.draft' not in p['path'] and f.file_hash(tmp_path/p['path'])==p['sha256']
    text=(tmp_path/result['output_files'][1]['path']).read_text()
    assert '00:00:01,000 --> 00:00:02,000' in text


def test_long_translation_needs_rewrite_not_retiming(tmp_path):
    s,m,rows,ret,_=setup_return(tmp_path,transform=lambda r:r[0].update(id_final='kata '*40))
    result=f.finalize(tmp_path,ret)
    assert any(x['code']=='id_final_line_length' for x in result['issues'])
    assert f.read_json(tmp_path/'schema.json')==s
    draft=(tmp_path/result['output_files'][1]['path']).read_text()
    assert draft.count('kata')==40 and '00:00:01,000 --> 00:00:02,000' in draft


def test_review_bound_to_exact_return_and_schema(tmp_path):
    s,_,_,ret,_=setup_return(tmp_path)
    r=reviewed(s,ret);r['return_sha256']='c'*64;f.write_json(tmp_path/'review.json',r)
    with pytest.raises(f.ContractError):f.finalize(tmp_path,ret)


def test_risks_cannot_be_removed_by_translator_flag(tmp_path):
    s=schema();s['cues'][0]['risk_flags']=['low_asr_confidence']
    s,_,_,ret,_=setup_return(tmp_path,s)
    f.write_json(tmp_path/'review.json',reviewed(s,ret))
    result=f.finalize(tmp_path,ret)
    assert any(i['code']=='low_asr_confidence' for i in result['issues'])


def test_reviewed_overlap_and_reading_speed_still_block(tmp_path):
    s,_,rows,ret,_=setup_return(tmp_path);r=reviewed(s,ret)
    c=s['cues'][0]
    r['cue_reviews'][c['block_uid']]=dict(start_ms=1000,end_ms=3200,tr_final=c['primary_text'],
        id_final='Defne, sini.',omit=False,reason='listened',listened=True)
    f.write_json(tmp_path/'review.json',r)
    assert any(x['code']=='subtitle_overlap' for x in f.finalize(tmp_path,ret)['issues'])


def test_gap_can_be_explicitly_non_speech_or_multiple_inserted_cues(tmp_path):
    s=schema();s['gaps']=[dict(issue_id='gap-0001',start_ms=4000,end_ms=6000)]
    s,_,rows,ret,_=setup_return(tmp_path,s);r=reviewed(s,ret)
    assert any(i['code']=='uncovered_speech' for i in f.qa(s,f.effective_rows(s,rows,r),r))
    r['gap_reviews']['gap-0001']=dict(not_speech=False,listened=True,reason='heard both',cues=[
        dict(start_ms=4000,end_ms=4800,tr_final='Evet.',id_final='Ya.'),
        dict(start_ms=5000,end_ms=5900,tr_final='Tamam.',id_final='Oke.')])
    effective=f.effective_rows(s,rows,r)
    assert len(effective)==4 and not f.qa(s,effective,r)
    r['gap_reviews']['gap-0001']=dict(not_speech=True,listened=True,reason='music only',cues=[])
    assert len(f.effective_rows(s,rows,r))==2


def test_copy_never_overwrites_different_source(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.write_bytes(b'one');b.write_bytes(b'two')
    with pytest.raises(f.ContractError):f.copy_verified(a,b)
    assert b.read_bytes()==b'two'


def test_srt_unicode_and_no_markup():
    assert f.wrap('Özlem, kamu nggak perlu khawatir.')=='Özlem, kamu nggak perlu khawatir.'
    for text in ['a\x00b','a --> b','<i>a</i>']:
        with pytest.raises(f.ContractError):f.wrap(text)


@pytest.mark.integration
def test_ffmpeg_preserves_delayed_audio_timeline(tmp_path):
    source=tmp_path/'offset.mkv';audio=tmp_path/'audio.wav'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=c=black:s=160x90:d=4',
        '-itsoffset','1.2','-f','lavfi','-i','sine=frequency=1000:duration=1',
        '-c:v','libx264','-c:a','pcm_s16le',str(source)],check=True,timeout=30)
    duration=f.extract_audio(source,audio)
    import numpy as np
    with wave.open(str(audio),'rb') as w: data=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
    first=int(np.flatnonzero(abs(data)>100)[0])/16
    assert 1180<=first<=1220
    assert 2180<=duration<=2250


def test_external_process_has_real_idle_timeout(tmp_path):
    with pytest.raises(TimeoutError):f.run_command([sys.executable,'-c','import time; time.sleep(5)'],idle_timeout=.2,timeout=3)


def test_notebook_embeds_exact_code_and_language_contract():
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'colab/Muhtemel_Ask.ipynb').read_text())
    sources=[''.join(c['source']) for c in nb['cells']]
    assert '%%writefile /content/ma_sub_colab.py\n'+(root/'src/mas/colab_flow.py').read_text() in sources
    for src in sources:
        if not src.startswith('%%') and src in [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']:
            compile(src,'cell','exec')
    assert all(not c.get('outputs') for c in nb['cells'])


def test_review_player_escapes_text_and_keeps_absolute_times():
    html=f.timing_clip_html(b'RIFF0000',500,dict(start_ms=1000,end_ms=2000,
                           tr_final='</script><img onerror=alert(1)>',id_final='Özlem'))
    assert '&lt;/script&gt;' in html and '<img onerror' not in html
    assert 't>=1000&&t<2000' in html and '500+a.currentTime*1000' in html


def test_interrupted_worker_reuses_completed_chunk_and_rejects_tampering(tmp_path,monkeypatch):
    import numpy as np
    root=tmp_path/'episode';root.mkdir();config=Path(__file__).resolve().parents[1]/'config/production'
    (root/'source.mp4').write_bytes(b'fake source for checkpoint test')
    source_sha=f.file_hash(root/'source.mp4')
    f.write_json(root/'source.json',dict(episode=15,sha256=source_sha,bytes=35,url=''))
    with wave.open(str(root/'audio.wav'),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(b'\0\0'*16000*730)
    f.write_json(root/'audio.json',dict(source_sha256=source_sha,sha256=f.file_hash(root/'audio.wav'),duration_ms=730000))
    state={'calls':0,'fail':True}
    class Model:
        def __init__(self,*args,**kwargs):pass
        def transcribe(self,audio,**kwargs):
            state['calls']+=1
            if state['fail'] and state['calls']==3:raise RuntimeError('simulated disconnect')
            word=types.SimpleNamespace(word='Merhaba.',start=1.,end=2.,probability=.95)
            segment=types.SimpleNamespace(id=0,words=[word],avg_logprob=-.1,no_speech_prob=.01,compression_ratio=1.,end=2.)
            return iter([segment]),None
    fw=types.ModuleType('faster_whisper');fw.WhisperModel=Model
    vad=types.ModuleType('faster_whisper.vad')
    vad.VadOptions=lambda **kwargs:kwargs
    vad.get_speech_timestamps=lambda audio,**kwargs:[dict(start=0,end=len(audio))]
    hub=types.ModuleType('huggingface_hub')
    hub.HfApi=lambda:types.SimpleNamespace(model_info=lambda *args,**kwargs:types.SimpleNamespace(sha='revision-test'))
    def model_download(*args,**kwargs):
        # A real T4 run failed with 80-vs-128 mel inputs when this file was omitted.
        assert 'preprocessor_config.json' in kwargs['allow_patterns']
        return 'model-test'
    hub.snapshot_download=model_download
    monkeypatch.setitem(sys.modules,'faster_whisper',fw);monkeypatch.setitem(sys.modules,'faster_whisper.vad',vad)
    monkeypatch.setitem(sys.modules,'huggingface_hub',hub)
    monkeypatch.setattr(f.importlib.metadata,'version',lambda p:'test-version')
    with pytest.raises(RuntimeError,match='simulated disconnect'):f.worker(root,config)
    checkpoint=root/'asr/chunk_0000.json';before=checkpoint.read_bytes()
    state['fail']=False
    f.worker(root,config)
    assert checkpoint.read_bytes()==before and state['calls']==7
    s=f.read_json(root/'schema.json')
    assert [w['start_ms'] for w in s['words']]==[1000,301000,601000]
    assert s['provenance']['model_revision']=='revision-test'
    assert 'chunk_cut_in_speech' in s['words'][1]['risk_flags']
    broken=f.read_json(checkpoint);broken['words'][0]['text']='Changed'
    f.write_json(checkpoint,broken)
    with pytest.raises(f.ContractError,match='Chunk checkpoint'):f.worker(root,config)


def test_vad_can_flag_early_end_without_automatically_extending_subtitles():
    ws=words()
    s=f.build_schema(15,'a'*64,6000,ws,[dict(start_ms=1000,end_ms=2400),dict(start_ms=2700,end_ms=4200)],{},'i',{})
    assert 'early_end_against_vad' in s['cues'][0]['risk_flags']
    assert s['cues'][0]['end_ms']==2000


def test_translation_cannot_add_unclaimed_timing_or_skip_foreign_content(tmp_path):
    s,m,rows,ret,_=setup_return(tmp_path)
    with zipfile.ZipFile(ret,'a') as z:z.writestr('translated_batch_001.jsonl',f.canonical(rows[0]))
    with pytest.raises(f.ContractError):f.read_return(ret,s,m)


def test_stock_phrase_is_preserved_for_review_unless_wide_audio_supports_it(monkeypatch):
    import numpy as np
    vad=types.ModuleType('faster_whisper.vad')
    vad.VadOptions=lambda **kwargs:kwargs
    vad.get_speech_timestamps=lambda audio,**kwargs:[dict(start=0,end=len(audio))]
    monkeypatch.setitem(sys.modules,'faster_whisper.vad',vad)
    class Model:
        def transcribe(self,*args,**kwargs):
            ws=[types.SimpleNamespace(word=t,start=s,end=e,probability=.8)
                for t,s,e in [('Altyazı',0.,0.),('M.K.',0.,.4)]]
            return iter([types.SimpleNamespace(id=0,words=ws,avg_logprob=-.1,
                        no_speech_prob=.1,compression_ratio=1.)]),None
    part=dict(index=0,start_ms=1000,end_ms=2000,unsafe_start=False,unsafe_end=False)
    reference=[dict(text='Bak.',start_ms=1000,end_ms=1400,probability=.9,risk_flags=[])]
    kept,rejected=f.speech_window_words(Model(),np.zeros(16000),part,reference,[])
    assert not kept and [w['text'] for w in rejected[0]['words']]==['Altyazı','M.K.']
    assert (rejected[0]['start_ms'],rejected[0]['end_ms'])==(1000,2000)
    reference[0]['text']='Altyazı M.K.'
    kept,rejected=f.speech_window_words(Model(),np.zeros(16000),part,reference,[])
    assert len(kept)==2 and not rejected  # Could be genuinely spoken; no blacklist deletion.
