import json,zipfile
from pathlib import Path
from ..hashing import sha256_json
IMMUTABLE=('block_uid','block_index','start_ms','end_ms')
def build_pack(path,episode,kind,records,schema_version='2.0'):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); schema={'schema_version':schema_version,'episode':episode,'block_count':len(records),'blocks':[{k:r[k] for k in IMMUTABLE} for r in records]}; schema_sha=sha256_json(schema); input_sha=sha256_json(records); manifest={'episode':episode,'kind':kind,'schema_version':schema_version,'schema_sha256':schema_sha,'input_sha256':input_sha,'record_count':len(records)}
 prompt='Correct Turkish text only. Preserve every immutable field exactly.' if kind=='TR' else 'Translate tr_text to natural Indonesian only. Preserve every immutable field exactly.'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)); z.writestr('schema.json',json.dumps(schema,ensure_ascii=False,indent=2)); z.writestr('records.json',json.dumps(records,ensure_ascii=False,indent=2)); z.writestr(f'{kind}_PROMPT.txt',prompt+'\n')
 return manifest
