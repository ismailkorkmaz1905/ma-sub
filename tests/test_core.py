import json,zipfile,pytest
from mas.packs.build import build_pack
from mas.packs.validate import validate_returned
from mas.errors import PackValidationError
from mas.subtitle.srt import render
R=[{'block_uid':'u1','block_index':1,'start_ms':1000,'end_ms':2000,'tr_text':'Defne','id_text':'Defne'}]
def test_pack_roundtrip(tmp_path):
 p=tmp_path/'p.zip'; m=build_pack(p,13,'TR',R); assert validate_returned(p,m,R)==R
def test_changed_uid_rejected(tmp_path):
 p=tmp_path/'p.zip'; m=build_pack(p,13,'TR',R)
 with zipfile.ZipFile(p) as z: f={n:z.read(n) for n in z.namelist()}
 r=json.loads(f['records.json']); r[0]['block_uid']='bad'; f['records.json']=json.dumps(r).encode()
 with zipfile.ZipFile(p,'w') as z:
  for n,b in f.items(): z.writestr(n,b)
 with pytest.raises(PackValidationError): validate_returned(p,m,R)
def test_srt(): assert '00:00:01,000 --> 00:00:02,000' in render(R,'id_text')
