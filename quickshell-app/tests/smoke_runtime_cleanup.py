#!/usr/bin/env python3
"""Exercise destructive pruning only against synthetic temporary builds."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile

script = Path(__file__).resolve().parents[1] / 'packaging/prune-runtimes.sh'
with tempfile.TemporaryDirectory(prefix='fractal-prune-') as temporary:
    root = Path(temporary)
    builds = []
    for number in range(6):
        build = root / f'0.1.0-{number:012x}-cpu'
        (build / 'bin').mkdir(parents=True)
        (build / 'bundle.json').write_text('{}')
        shutil.copy('/usr/bin/sleep', build / 'bin/fractal-search')
        builds.append(build)
    outside = root / 'unrelated'
    outside.mkdir()
    (root / '0.1.0-aaaaaaaaaaaa-cpu').symlink_to(outside, target_is_directory=True)
    current, previous, busy, cuda, obsolete, older = builds
    process = subprocess.Popen([str(busy / 'bin/fractal-search'), '60'])
    try:
        def prune(prior):
            subprocess.run([str(script), str(root), str(current), str(prior), str(cuda / 'bin')], check=True)
        prune(previous)
        assert all(p.exists() for p in (current, previous, busy, cuda, outside))
        assert not obsolete.exists() and not older.exists()
        prune(current)
        assert previous.exists()
    finally:
        process.terminate()
        process.wait()
    prune(current)
    assert not busy.exists()
    assert previous.exists() and cuda.exists()
print('Runtime cleanup: retention, reinstall, live executable, CUDA directory and symlink confinement passed')
