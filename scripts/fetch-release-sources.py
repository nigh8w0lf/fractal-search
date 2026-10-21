#!/usr/bin/env python3
"""Fetch pinned covered source for release redistribution; no model downloads."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'target/release-sources'


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / 'quickshell-app/packaging/licenses/sources.json').read_text())
    for record in records:
        target = CACHE / record['filename']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise SystemExit(f'Covered source checksum mismatch: {target}; inspect/remove it explicitly')
            print(f'Verified cached covered source: {record["filename"]}')
            continue
        if not record['url'].startswith('https://'):
            raise SystemExit('Covered source downloads require HTTPS')
        print(f'Downloading covered source: {record["url"]}', flush=True)
        with urllib.request.urlopen(record['url'], timeout=120) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise SystemExit(f'Covered source size/checksum mismatch: {record["filename"]}')
        with target.open('xb') as output:
            output.write(data)


if __name__ == '__main__':
    main()
