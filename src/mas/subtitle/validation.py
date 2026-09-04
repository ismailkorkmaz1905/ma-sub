def qc(records,key='id_text',max_cps=20.0,min_ms=700,max_ms=7000):
 issues=[]; prev=-1
 for i,r in enumerate(records):
  s,e=int(r['start_ms']),int(r['end_ms']); text=str(r.get(key,'')).strip(); d=e-s
  if e<=s: issues.append({'block':i+1,'code':'invalid_timing'})
  if s<prev: issues.append({'block':i+1,'code':'overlap'})
  if d<min_ms: issues.append({'block':i+1,'code':'too_short'})
  if d>max_ms: issues.append({'block':i+1,'code':'too_long'})
  if not text: issues.append({'block':i+1,'code':'missing_translation'})
  elif d>0 and len(text.replace('\n',''))/(d/1000)>max_cps: issues.append({'block':i+1,'code':'high_cps'})
  prev=e
 return issues
