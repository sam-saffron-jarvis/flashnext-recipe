#!/usr/bin/env python3
"""Sequential pinned GGUF acquisition; one curl transfer at a time, SHA256 verified."""
import hashlib, json, pathlib, sys, os
root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root/'control'))
os.environ['STRATA_DL_CONNECTIONS']='4'
import setup as strata_setup
meta = json.loads((root/'downloads/hf-ista-metadata.json').read_text())
repo='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'
rev='ed59f92082b1e93c0e96d60a8b11aab089b52f09'
records=[]
for f in meta['siblings']:
    name=f['rfilename']
    if not name.startswith('IQ3_XXS/') or not name.endswith('.gguf'): continue
    path=root/'models'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != f['size']:
        print('Downloading', name, f['size'], flush=True)
        # The official setup download helper is CPU/network only. Four ranged
        # workers for this file; shards are still acquired sequentially.
        partial=path.with_name(path.name+'.part')
        if path.exists() and not partial.exists(): path.rename(partial)
        strata_setup.download(f'https://huggingface.co/{repo}/resolve/{rev}/{name}', path, name)
    h=hashlib.sha256()
    with path.open('rb') as src:
        while b:=src.read(8<<20): h.update(b)
    assert path.stat().st_size == f['size'], (name,'size mismatch')
    assert h.hexdigest() == f['lfs']['sha256'], (name,'hash mismatch')
    r={'repo':repo,'revision':rev,'path':name,'bytes':f['size'],'sha256':h.hexdigest()}
    records.append(r); print('VERIFIED',r,flush=True)
(root/'downloads/model-verification.json').write_text(json.dumps(records,indent=2)+'\n')
