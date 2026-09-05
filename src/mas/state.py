import json,os,tempfile
from pathlib import Path
from datetime import datetime,timezone
from . import __version__,SCHEMA_VERSION,RULES_VERSION
def load(path,episode):
 p=Path(path)
 if not p.exists(): return {'episode':episode,'pipeline_version':__version__,'schema_version':SCHEMA_VERSION,'rules_version':RULES_VERSION,'stages':{},'created_at':datetime.now(timezone.utc).isoformat()}
 d=json.loads(p.read_text(encoding='utf-8'))
 if d.get('episode')!=episode: raise ValueError('state episode mismatch')
 return d
def save(path,state):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); state['updated_at']=datetime.now(timezone.utc).isoformat(); fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.state-',text=True); os.close(fd); Path(tmp).write_text(json.dumps(state,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,p)
def set_stage(state_path,state,name,status,**details):
 if status not in {'pending','running','pass','blocked','failed'}: raise ValueError(f'invalid stage status: {status}')
 state.setdefault('stages',{})[name]={'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),**details}; save(state_path,state)
def reusable(state,name,input_sha,config_sha,verify):
 s=state.get('stages',{}).get(name,{}); return s.get('status')=='complete' and s.get('input_sha256')==input_sha and s.get('config_sha256')==config_sha and all(verify(x) for x in s.get('outputs',[]))
def invalidate_after(state,ordered,name):
 if name in ordered:
  for x in ordered[ordered.index(name)+1:]: state.get('stages',{}).pop(x,None)
