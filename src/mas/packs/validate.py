import json,zipfile
from ..errors import PackValidationError
IMMUTABLE=('block_uid','block_index','start_ms','end_ms')
def validate_returned(path,expected_manifest,expected_records):
 with zipfile.ZipFile(path) as z: m=json.loads(z.read('manifest.json')); records=json.loads(z.read('records.json'))
 for k in ('episode','kind','schema_version','schema_sha256','input_sha256','record_count'):
  if m.get(k)!=expected_manifest.get(k): raise PackValidationError(f'manifest invariant failed: {k}: expected {expected_manifest.get(k)!r}, got {m.get(k)!r}')
 if len(records)!=len(expected_records): raise PackValidationError(f'record count changed: expected {len(expected_records)}, got {len(records)}')
 seen=set()
 for i,(a,b) in enumerate(zip(expected_records,records)):
  uid=b.get('block_uid')
  if uid in seen: raise PackValidationError(f'duplicate block_uid at record {i}: {uid}')
  seen.add(uid)
  for k in IMMUTABLE:
   if b.get(k)!=a.get(k): raise PackValidationError(f'immutable field changed at record {i} ({a.get("block_uid")}): {k}: expected {a.get(k)!r}, got {b.get(k)!r}')
 return records
