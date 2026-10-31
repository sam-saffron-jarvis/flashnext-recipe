#!/usr/bin/env bash
# CPU/network-only preparation (MTP draft tensors, native pack, draft runtime).
# Never invokes the engine, CUDA, setup main(), or a server.
# Usage: prepare-cpu.sh STRATA_ROOT   (after download_models.py has verified the GGUFs)
set -euo pipefail
cd -- "${1:?usage: prepare-cpu.sh STRATA_ROOT}"
ROOT=$PWD
export HOME="$ROOT/home" XDG_CACHE_HOME="$ROOT/cache"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" STRATA_DL_CONNECTIONS=4
export STRATA_GGUF_PY="$ROOT/tools/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d/gguf-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="$ROOT/control/python/bin/python3"
# Called only once the sequential GGUF downloader has exited successfully.
test -f downloads/model-verification.json
"$PY" - <<'PY'
import json,urllib.request,pathlib
repo='Qwen/Qwen3.8-Flash-Next'; rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
u=f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true'
with urllib.request.urlopen(u,timeout=60) as r: b=r.read()
pathlib.Path('downloads/hf-mtp-source-metadata.json').write_bytes(b)
x=json.loads(b); assert x['sha']==rev
print('Recorded pinned MTP checkpoint metadata',rev,flush=True)
PY
"$PY" control/tools/mtp_fetch.py fetch --out models/mtp
"$PY" control/tools/mtp_fetch.py verify --out models/mtp
"$PY" control/tools/iq_pack.py --gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out models/pack/iq3_xxs
"$PY" control/tools/mtp_pack.py --src models/mtp --experts q2_0 --out models/mtp/mtp-q2_0.gguf
"$PY" control/tools/mtp_rt.py --gguf models/mtp/mtp-q2_0.gguf --out models/mtp/rt
cp control/data/draft_vocab.bin models/mtp/rt/draft_vocab.bin
"$PY" - <<'PY'
import pathlib,hashlib,json
root=pathlib.Path('.'); out=[]
for base in ['models/pack','models/mtp/rt']:
 for p in sorted(root.joinpath(base).rglob('*')):
  if p.is_file():
   with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
   out.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h})
p=pathlib.Path('models/mtp/mtp-q2_0.gguf')
with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
out.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h})
pathlib.Path('downloads/prepared-artifacts.json').write_text(json.dumps(out,indent=2)+'\n')
print('PREPARATION COMPLETE; no engine or GPU was started.',flush=True)
PY
