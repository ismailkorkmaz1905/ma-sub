def ts(ms):
 h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000); return f'{h:02}:{m:02}:{s:02},{ms:03}'
def render(records,key):
 out=[]
 for i,r in enumerate(records,1):
  text=str(r.get(key,'')).strip()
  if not text: raise ValueError(f'empty subtitle text at block {i}')
  out += [str(i),f"{ts(int(r['start_ms']))} --> {ts(int(r['end_ms']))}",text,'']
 return '\n'.join(out)
