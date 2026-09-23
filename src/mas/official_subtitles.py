"""Publisher captions -> ChatGPT translation -> SRT. No ASR, GPU or paid API.

The scheduled ChatGPT task performs translation and Drive I/O using its connectors.
This module only downloads public caption data and validates/resumes the files.
"""
from __future__ import annotations
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request
from urllib.parse import urljoin, urlparse
import zipfile

from mas import colab_flow as f

INDEX = 'https://www.showtv.com.tr/dizi/tanitim/muhtemel-ask/3072'
VERSION = 'official-captions-1'
NOTES = '''# Publisher captions, not ASR
The Turkish text and cue times come from Show TV's published Turkish WebVTT.
No ASR, second recognizer or audio listening has occurred. Preserve original cue
IDs, order and times. Use the unchanged production instructions for translation.
Translate sound/music descriptions too; do not omit spoken content or add speech.
Use concise natural Indonesian within the per-cue character budget; never stretch
or shift timestamps to fit text. Flag ambiguity or an impossible budget honestly.
These captions belong ONLY to the source media identified in source.json. They
must not be declared synchronized to a YouTube or Drive edit without comparison.
'''

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.links=[];self.players=[];self.metadata=[];self._json=False;self._body=''
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag=='a' and attrs.get('href'):self.links.append(attrs['href'])
        if attrs.get('data-hope-video'):self.players.append(f.strict_json(attrs['data-hope-video']))
        if tag=='script' and attrs.get('type')=='application/ld+json':self._json=True;self._body=''
    def handle_data(self, data):
        if self._json:self._body+=data
    def handle_endtag(self, tag):
        if tag=='script' and self._json:
            self.metadata.append(f.strict_json(self._body));self._json=False


def get(url, limit=4_000_000):
    """Bounded public publisher reads only; no media/geoblock workarounds."""
    if urlparse(url).scheme!='https' or urlparse(url).hostname not in {'www.showtv.com.tr','vmcdn.ciner.com.tr'}:
        raise f.ContractError('Unexpected publisher URL')
    request=urllib.request.Request(url,headers={'User-Agent':'ma-sub/1.0 public-caption-reader'})
    with urllib.request.urlopen(request,timeout=30) as response:
        if urlparse(response.url).hostname not in {'www.showtv.com.tr','vmcdn.ciner.com.tr'}:
            raise f.ContractError('Unexpected publisher redirect')
        data=response.read(limit+1)
    if len(data)>limit:raise f.ContractError('Publisher response exceeds size limit')
    return data


def discover(episode):
    page=Page(get(INDEX).decode('utf-8'))
    pattern=rf'/dizi/tum_bolumler/muhtemel-ask-sezon-1-bolum-{episode}-izle/\d+$'
    urls={urljoin(INDEX,p) for p in page.links if re.fullmatch(pattern,p)}
    if len(urls)>1:raise f.ContractError('Ambiguous episode links')
    return next(iter(urls),None)


def milliseconds(stamp):
    parts=stamp.split(':')
    if len(parts)==2:parts.insert(0,'0')
    if len(parts)!=3 or not re.fullmatch(r'\d{2}\.\d{3}',parts[2]):
        raise f.ContractError('Invalid WebVTT timestamp')
    h,m=int(parts[0]),int(parts[1]);s,ms=map(int,parts[2].split('.'))
    if not 0<=m<60 or not 0<=s<60:raise f.ContractError('Invalid WebVTT clock')
    return ((h*60+m)*60+s)*1000+ms


def parse_vtt(data):
    text=data.decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n').replace('\ufeff','')
    if not text.startswith('WEBVTT'):raise f.ContractError('Not a WebVTT file')
    cues=[];sha=hashlib.sha256(data).hexdigest()
    for block in re.split(r'\n[ \t]*\n',text):
        lines=block.strip().splitlines()
        if not lines or lines[0].startswith(('WEBVTT','NOTE','STYLE','REGION')):continue
        index=next((i for i,line in enumerate(lines) if '-->' in line),None)
        if index is None:raise f.ContractError('Unrecognized caption block')
        match=re.fullmatch(r'(\S+)\s+-->\s+(\S+)(?:\s+.*)?',lines[index])
        if not match:raise f.ContractError('Invalid caption timing')
        start,end=map(milliseconds,match.groups())
        if start>=end or cues and start<cues[-1]['end_ms']:
            raise f.ContractError('Invalid/overlapping publisher cue; inspect before translating')
        # Remove WebVTT presentation tags only; preserve all visible words.
        value=html.unescape(re.sub(r'<[^>]*>','', '\n'.join(lines[index+1:])))
        value=f.safe_text(value)
        cues.append(dict(block_uid=f'{sha[:12]}-{len(cues)+1:05d}',start_ms=start,end_ms=end,
                         primary_text=value,risk_flags=[]))
    if not cues:raise f.ContractError('Publisher captions are empty')
    return cues


def prepare(root, episode, config, page_url=None):
    root=Path(root);config=Path(config)
    if (root/'schema.json').exists():
        schema=f.read_json(root/'schema.json')
        if schema['episode']!=episode or schema['version']!=VERSION:
            raise f.ContractError('Different episode/workflow already in folder')
        if f.file_hash(root/'original.tr.vtt')!=schema['source_sha256']:
            raise f.ContractError('Saved publisher captions changed')
        if f.digest(f.read_json(root/'source.json'))!=schema['provenance']['source_record_sha256']:
            raise f.ContractError('Saved source binding changed')
        f.make_pack(root,schema,(config/'TRANSLATION_INSTRUCTIONS.md').read_text(),
                    batch_size=60,evidence_notes=NOTES)
        return dict(status='READY',episode=episode,cues=len(schema['cues']))
    page_url=page_url or discover(episode)
    if not page_url:return dict(status='WAITING_FOR_EPISODE',episode=episode)
    if not re.fullmatch(rf'https://www\.showtv\.com\.tr/dizi/tum_bolumler/muhtemel-ask-sezon-1-bolum-{episode}-izle/\d+',page_url):
        raise f.ContractError('Not the requested official full episode URL')
    page=Page(get(page_url).decode('utf-8'))
    videos=[j for j in page.metadata if isinstance(j,dict) and j.get('@type')=='VideoObject']
    episodes=[j for j in page.metadata if isinstance(j,dict) and j.get('@type')=='TVEpisode']
    if len(videos)!=1 or len(episodes)!=1 or str(episodes[0].get('episodeNumber'))!=str(episode):
        raise f.ContractError('Publisher episode identity is ambiguous')
    if len(page.players)!=1:raise f.ContractError('Publisher player identity is ambiguous')
    tracks=[s for s in page.players[0].get('subtitles',[]) if s.get('srclang')=='tr']
    if not tracks:return dict(status='WAITING_FOR_TURKISH_CAPTIONS',episode=episode,page_url=page_url)
    if len(tracks)!=1:raise f.ContractError('Multiple Turkish caption tracks')
    video=videos[0]
    content_url=video['contentUrl'];caption_url=tracks[0]['src']
    stem=Path(urlparse(content_url).path).stem.split('_')[0]
    if not re.fullmatch('[A-Fa-f0-9]{32}',stem) or not Path(urlparse(caption_url).path).name.startswith(stem+'_'):
        raise f.ContractError('Caption and media asset IDs differ')
    duration=re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',video['duration'])
    if not duration:raise f.ContractError('Unknown media duration')
    h,m,s=[int(x or 0) for x in duration.groups()];duration_ms=((h*60+m)*60+s)*1000
    data=get(caption_url);cues=parse_vtt(data)
    if duration_ms<1_800_000 or cues[-1]['end_ms']>duration_ms+1000 or cues[-1]['end_ms']<duration_ms-300_000:
        raise f.ContractError('Captions do not cover the expected full-episode timeline')
    import yaml
    names=yaml.safe_load((config/'names.yaml').read_text());terms=yaml.safe_load((config/'religious_terms.yaml').read_text())
    instructions=(config/'TRANSLATION_INSTRUCTIONS.md').read_text()
    source=dict(episode=episode,page_url=page_url,media_url=content_url,caption_url=caption_url,
                media_asset_id=stem,duration_ms=duration_ms,caption_sha256=hashlib.sha256(data).hexdigest(),
                allowed_video_countries=page.players[0].get('geoChecker',{}).get('allowedCountries',[]),
                timing_authority='publisher WebVTT; unchanged',audio_sync_verified=False,
                other_video_edits_compatible=False)
    glossary=dict(canonical_names=names['canonical_names'],forbidden_name_variants=names['forbidden_variants'],
                  source_name_variants=names['source_variants'],religious_terms=terms)
    schema=dict(version=VERSION,scope='full_episode',episode=episode,source_sha256=source['caption_sha256'],duration_ms=duration_ms,
                instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest(),glossary=glossary,cues=cues,
                provenance=dict(source_kind='publisher_captions',source_record_sha256=f.digest(source)))
    f.atomic_bytes(root/'original.tr.vtt',data);f.write_json(root/'source.json',source)
    f.make_pack(root,schema,instructions,batch_size=60,evidence_notes=NOTES)
    return dict(status='READY',episode=episode,cues=len(cues),source=source)


def pending(root):
    root=Path(root);schema=f.read_json(root/'schema.json');manifest=f.read_json(root/'manifest.json')
    for batch in manifest['batches']:
        path=root/'translations'/('translated_'+batch['filename'])
        if not path.exists():return batch['filename']
        validate_batch(path,schema,batch)
    return None


def validate_batch(path,schema,batch):
    rows=[f.strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if [r.get('block_uid') for r in rows]!=batch['block_uids']:raise f.ContractError('Translation UID/order mismatch')
    for r in rows:
        if set(r)!=f.OUTPUT_KEYS or r['schema_sha256']!=f.digest(schema) or type(r['review_required']) is not bool or not isinstance(r['note'],str):
            raise f.ContractError('Invalid translated record')
        f.safe_text(r['tr_final']);f.safe_text(r['id_final'])
    return rows


def finish(root):
    root=Path(root);schema=f.read_json(root/'schema.json');manifest=f.read_json(root/'manifest.json')
    if schema['version']!=VERSION or f.digest(schema)!=manifest['schema_sha256']:raise f.ContractError('Schema changed')
    if f.file_hash(root/'original.tr.vtt')!=schema['source_sha256']:raise f.ContractError('Source captions changed')
    if f.digest(f.read_json(root/'source.json'))!=schema['provenance']['source_record_sha256']:raise f.ContractError('Source record changed')
    original=parse_vtt((root/'original.tr.vtt').read_bytes())
    if schema.get('scope')=='sample':
        original=[c for c in original if schema['sample_start_ms']<=c['start_ms'] and c['end_ms']<=schema['sample_end_ms']]
    elif schema.get('scope')!='full_episode':raise f.ContractError('Unknown output scope')
    if original!=schema['cues']:raise f.ContractError('Publisher cues or timings changed')
    translations=[]
    for batch in manifest['batches']:
        translations.extend(validate_batch(root/'translations'/('translated_'+batch['filename']),schema,batch))
    rows=[dict(c,**{k:v for k,v in r.items() if k!='block_uid'}) for c,r in zip(schema['cues'],translations)]
    if [r['block_uid'] for r in rows]!=[c['block_uid'] for c in schema['cues']]:raise f.ContractError('Incomplete translation')
    issues=[]
    for row in rows:
        uid=row['block_uid']
        if row['review_required']:issues.append(dict(block_uid=uid,code='translation_review',note=row['note']))
        for language in ['tr_final','id_final']:
            text=f.safe_text(row[language])
            try:f.wrap(text)
            except f.ContractError:issues.append(dict(block_uid=uid,code=language+'_line_length'))
            if len(text)>20*(row['end_ms']-row['start_ms'])/1000:issues.append(dict(block_uid=uid,code=language+'_reading_speed'))
            for aliases in schema['glossary']['forbidden_name_variants'].values():
                if any(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',text,re.I) for a in aliases):
                    issues.append(dict(block_uid=uid,code=language+'_name'))
        if 'allah' in row['primary_text'].casefold() and 'allah' not in row['id_final'].casefold():
            issues.append(dict(block_uid=uid,code='allah_missing'))
    out=root/'output';out.mkdir(exist_ok=True)
    files=[]
    for lang,key in [('TR','tr_final'),('ID','id_final')]:
        name=f'Muhtemel Ask {schema["episode"]}.Bolum_SHOWTV_{lang}'+('.sample' if schema['scope']=='sample' else '')+('.draft' if issues else '')+'.srt'
        f.atomic_bytes(out/name,f.srt(rows,key,draft=bool(issues)).encode())
        path=out/name;files.append(dict(name=name,bytes=path.stat().st_size,sha256=f.file_hash(path)))
    report=dict(status='DRAFT_REVIEW_REQUIRED' if issues else 'TRANSLATED_WITH_PUBLISHER_TIMES',
                episode=schema['episode'],scope=schema['scope'],cue_count=len(rows),schema_sha256=f.digest(schema),
                source=f.read_json(root/'source.json'),issues=issues,files=files,
                audio_sync_verified=False,warning='Only the identified Show TV edit. No claim of YouTube sync or full listening.')
    f.write_json(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','pending','finish']);p.add_argument('--root',required=True)
    p.add_argument('--episode',type=int,default=15);p.add_argument('--page-url')
    p.add_argument('--config',default='config/production');a=p.parse_args()
    result=prepare(a.root,a.episode,a.config,a.page_url) if a.action=='prepare' else pending(a.root) if a.action=='pending' else finish(a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
