#!/usr/bin/env python3
"""CPU-only installed-artifact/config checks. Never starts an engine or server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'control'))
sys.path.insert(0, str(ROOT / 'control/tools'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full-model-hashes', action='store_true', help='also reread both large GGUF files')
    args = ap.parse_args()
    from strata_control import signing, engine_component, profiles
    from serve.server import Service, MockEngine, ByteTokenizer, clean_shared_defaults
    from serve.frontend import ChatTemplate
    from strata_tokenizer import Tokenizer
    report = {'gpu_used': False, 'checks': []}
    down = ROOT / 'downloads'
    feed_bytes = (down / 'strata-feed.json').read_bytes()
    key = signing.verify_bytes(signing.load_trusted_keys(), feed_bytes,
                               json.loads((down / 'strata-feed.json.sig').read_text()))
    report['feed_key_id'] = key
    feed = json.loads(feed_bytes)
    for kind, name in [('control', 'strata-control-0.1.55-linux-x86_64.tar.gz'),
                       ('engine', 'strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz')]:
        pkg = next(p for p in feed[kind]['packages'] if p['url'] == name)
        path = down / name
        assert path.stat().st_size == pkg['size'] and digest(path) == pkg['sha256'], name
        if kind == 'engine':
            component = engine_component.read_manifest(ROOT / 'components/engine/0.1.55-linux-x86_64-cuda-cuda13')
            assert digest(component.root / 'engine-component.json') == pkg['manifest_sha256']
            engine_component.verify(component, full=True)
        report['checks'].append({'archive': name, 'bytes': pkg['size'], 'sha256': pkg['sha256']})
    meta = json.loads((down / 'hf-ista-metadata.json').read_text())
    records = json.loads((down / 'model-verification.json').read_text())
    assert len(records) == 2
    for rec in records:
        remote = next(f for f in meta['siblings'] if f['rfilename'] == rec['path'])
        assert rec['revision'] == meta['sha']
        assert (rec['bytes'], rec['sha256']) == (remote['size'], remote['lfs']['sha256'])
        path = ROOT / 'models' / rec['path']
        assert path.stat().st_size == rec['bytes']
        if args.full_model_hashes:
            assert digest(path) == rec['sha256']
    prepared = json.loads((down / 'prepared-artifacts.json').read_text())
    for rec in prepared:
        path = ROOT / rec['path']
        assert path.stat().st_size == rec['bytes'] and digest(path) == rec['sha256'], rec['path']
    # Full MTP tensor hashes checked against both inventory ranges and Strata's pins.
    from mtp_fetch import SHA256
    inv = json.loads((ROOT / 'models/mtp/mtp-inventory.json').read_text())
    assert 'de4b8e4d43b917e7706784d8bb445c9af86a3540' in inv['repo']
    tensors = json.loads((ROOT / 'models/mtp/mtp-manifest.json').read_text())
    assert len(tensors) == 31
    for rec in tensors:
        path = ROOT / 'models/mtp' / rec['file']
        assert path.stat().st_size == rec['bytes'] and digest(path) == rec['sha256'] == SHA256[rec['name']], rec['name']
    cfg = profiles.read_run_config(ROOT / 'strata-iq3-128k.json')
    fallback = profiles.read_run_config(ROOT / 'strata-iq3-64k.json')
    assert fallback['args'][fallback['args'].index('--max-context') + 1] == '65536'
    assert cfg['host'] == '127.0.0.1' and cfg['port'] == 18199 and cfg['parallel'] == 1
    clean_shared_defaults({'reasoning_effort': cfg['reasoning_effort'], 'reasoning_budget_tokens': cfg['reasoning_budget_tokens']})
    a = cfg['args']
    for flag in ['--pack', '--native', '--ple-gguf', '--mtp', '--expert-profile']:
        p = Path(a[a.index(flag) + 1]); assert p.exists() and p.is_relative_to(ROOT), p
    assert a[a.index('--max-context') + 1] == '131072'
    tp = Path(cfg['tokenizer'])
    vocab = json.loads((tp / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items(): tokens[int(i)] = t
    merges = (tp / 'merges.txt').read_text().splitlines()
    if merges and merges[0].startswith('#'): merges = merges[1:]
    types = json.loads((tp / 'token_type.json').read_text())
    tc = json.loads((tp / 'tokenizer.json').read_text())
    tok = Tokenizer(tokens, merges, types, tc.get('pre') or 'qwen35', tc.get('special_ids') or None)
    for text in ['READY', 'Tools: {"path":"test.py"}', 'Sydney café — 日本語']:
        assert tok.decode(tok.encode(text)) == text
    template = ChatTemplate(tp / 'chat_template.jinja')
    prompt = template.render([{'role': 'user', 'content': 'Check thinking.'}], reasoning_effort='medium')
    assert '<think>' in prompt and not prompt.rstrip().endswith('</think>')
    svc = Service(MockEngine(ByteTokenizer(), 'unused'), ByteTokenizer(), template)
    svc.reasoning_budget_tokens = cfg['reasoning_budget_tokens']
    assert svc.reasoning_budget({}) == 2048 and svc.reasoning_budget({'thinking_budget_tokens': 128}) == 128
    assert svc.reasoning_budget({'reasoning_budget_tokens': 0}) is None
    from gguf_reader import GGUFFile
    g = GGUFFile(a[a.index('--native') + 1])
    expert_bytes = sum(t.expected_bytes() or 0 for t in g.tensors if 'ffn_gate_exps' in t.name or 'ffn_up_exps' in t.name or 'ffn_down_exps' in t.name)
    report['expert_bytes_in_shard1'] = expert_bytes
    report['model_architecture'] = g.metadata.get('general.architecture')
    report['model_context_length'] = g.metadata.get('qwen4exp.context_length')
    import benchmark
    report['benchmark_prompt_calibration'] = []
    for target in [2048, 8192, 16384, 32768, 54000]:
        text, lines = benchmark.facts(target, 'Reply with exactly: READY')
        rendered = template.render([{'role': 'user', 'content': text}], enable_thinking=False)
        actual = len(tok.encode(rendered, parse_special=True))
        assert actual + 1000 <= 65536, (target, actual, 'would not fit 64K fallback')
        report['benchmark_prompt_calibration'].append({'target': target, 'lines': lines, 'actual_template_tokens': actual})
    report['checks'] += ['GGUF provenance/size checks; full SHA256 already checked by acquisition' + (' and rechecked now' if args.full_model_hashes else ''),
                         'all generated artifacts and 31 MTP tensor SHA256s', 'config paths/defaults',
                         'tokenizer UTF-8 round trips', 'real model template thinking enabled', 'bounded-thinking defaults/overrides']
    s = socket.socket()
    try:
        s.bind(('127.0.0.1', 18199))
        report['port_18199_free'] = True
    finally:
        s.close()
    (ROOT / 'logs/cpu-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
