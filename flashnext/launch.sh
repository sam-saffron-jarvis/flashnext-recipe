#!/bin/sh
set -eu
root=${FLASHNEXT_ROOT:?set FLASHNEXT_ROOT to the retained or adopted runtime root}
test "$(git -C "$root/runtime/llama.cpp" rev-parse HEAD)" = 74bca3950054eed1795f525cb9cf06bfe796492d
for shard in 00001 00002 00003; do
    test -s "$root/models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-$shard-of-00003.gguf"
done
export LD_LIBRARY_PATH="$root/runtime/llama.cpp/build/bin:$root/runtime/tools/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/runtime/llama.cpp/build/bin/llama-server" \
 --model "$root/models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" \
 --alias flashnext --host 127.0.0.1 --port 18099 --offline --no-webui \
 -fa on -ngl 99 -t 24 -C 0xFFFF5555 --cpu-strict 1 -c 65536 -np 1 \
 --n-cpu-moe 38 -ub 1024 -ctk f16 -ctv f16 --jinja
