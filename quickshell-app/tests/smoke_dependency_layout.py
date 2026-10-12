#!/usr/bin/env python3
"""Guard targeted build isolation and the ICU-only pinned dependency patch."""
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
host = next(line.split(': ', 1)[1] for line in subprocess.check_output(
    ['rustc', '-vV'], text=True).splitlines() if line.startswith('host: '))
metadata = json.loads(subprocess.check_output([
    'cargo', 'metadata', '--locked', '--offline', '--format-version', '1',
    '--filter-platform', host], cwd=ROOT, text=True))
packages = {p['id']: p for p in metadata['packages']}
nodes = {n['id']: n for n in metadata['resolve']['nodes']}
members = {packages[i]['name']: i for i in metadata['workspace_members']}
assert set(members) == {'fractal-search-core', 'fractal-search',
                        'fractal-folder-picker', 'fractal-search-mcp'}, members


def closure(root):
    visited, pending = set(), [root]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(d['pkg'] for d in nodes[current]['deps'])
    return {packages[i]['name'] for i in visited}


picker = closure(members['fractal-folder-picker'])
assert not any(name.startswith(('lance', 'arrow', 'datafusion', 'ort', 'tokenizers',
                                'rmcp', 'fractal-search-core')) for name in picker), picker
for owner, forbidden in [('fractal-search', {'zbus', 'rmcp', 'base64'}),
                         ('fractal-search-mcp', {'zbus', 'notify', 'toml', 'fractal-search'})]:
    direct = {packages[d['pkg']]['name'] for d in nodes[members[owner]]['deps']}
    assert not direct & forbidden, (owner, direct & forbidden)
active = {packages[i]['name'] for i in nodes}
assert not active & {'jieba-rs', 'lindera', 'lindera-dictionary'}, active
index = next(n for i, n in nodes.items() if packages[i]['name'] == 'lance-index')
assert not set(index['features']) & {'tokenizer-jieba', 'tokenizer-lindera', 'jieba-rs', 'lindera'}
lancedb = next(p for p in packages.values() if p['name'] == 'lancedb')
vendor = ROOT / 'vendor/lancedb'
assert lancedb['version'] == '0.37.1'
assert Path(lancedb['manifest_path']) == vendor / 'Cargo.toml'
upstream = tomllib.loads((vendor / 'Cargo.toml.upstream').read_text())
patched = tomllib.loads((vendor / 'Cargo.toml').read_text())
expected = upstream['dependencies']['lance-index']
assert expected.pop('features') == ['tokenizer-jieba', 'tokenizer-lindera']
expected['default-features'] = False
assert patched == upstream, 'The local patch changed more than the tokenizer feature request'
hashes = json.loads((vendor / 'UPSTREAM-SHA256.json').read_text())
actual = {str(p.relative_to(vendor)) for name in ['src', 'examples', 'tests']
          for p in (vendor / name).rglob('*') if p.is_file()}
assert actual == set(hashes), 'Vendored upstream file set changed'
for relative, expected_hash in hashes.items():
    assert hashlib.sha256((vendor / relative).read_bytes()).hexdigest() == expected_hash, relative
print('Dependencies: isolated picker, separate binary ownership, ICU-only feature graph, '
      'pinned local patch and unchanged upstream Rust source passed')
