#!/usr/bin/env python3
"""Collect exact resolved dependency notices; fail on missing attribution inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
LEGAL = ROOT / 'quickshell-app/packaging/licenses'


def output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def license_files(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file()
                  and p.name.lower().startswith(('license', 'licence', 'copying', 'copyright', 'notice')))


def source_archive(directory, target, name):
    def normalized(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        return info

    # Stable attribution must not create a new runtime identity on every packaging run.
    with target.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                archive.add(directory, arcname=name, filter=normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    destination = bundle / 'third-party'
    if destination.exists():
        raise SystemExit(f'Refusing to overwrite {destination}')
    host = next(line.split(': ', 1)[1] for line in output('rustc', '-vV').splitlines() if line.startswith('host: '))
    command = ['cargo', 'metadata', '--locked', '--offline', '--format-version', '1', '--filter-platform', host]
    if args.cuda:
        command += ['--features', 'fractal-search/cuda,fractal-search-mcp/cuda']
    metadata = json.loads(output(*command))
    overrides = json.loads((LEGAL / 'overrides.json').read_text())
    provenance = json.loads((LEGAL / 'provenance.json').read_text())
    for record in provenance:
        data = (LEGAL / record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise SystemExit(f"License provenance checksum mismatch: {record['path']}")
    entries = []
    for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        if package['id'] in metadata['workspace_members']:
            continue
        key = package['name'] + '-' + package['version']
        directory = Path(package['manifest_path']).parent
        files = license_files(directory)
        if package.get('license_file'):
            explicit = directory / package['license_file']
            if not explicit.is_file():
                raise SystemExit(f'Missing declared license file for {key}')
            files = sorted(set(files + [explicit]))
        if not package.get('license') and not package.get('license_file'):
            raise SystemExit(f'Undeclared license for {key}')
        copied = []
        for source in files:
            relative = source.relative_to(directory)
            target = destination / 'crates' / key / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(str(target.relative_to(bundle)))
        for relative in overrides.get(key, []):
            source = LEGAL / relative
            if relative not in {r['path'] for r in provenance}:
                raise SystemExit(f'Unverified license override: {relative}')
            target = destination / 'crates' / key / ('upstream-' + source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(str(target.relative_to(bundle)))
        if not copied:
            raise SystemExit(f'Missing license/notice text for {key}; add a versioned upstream override')
        entry = {k: package.get(k) for k in ['name', 'version', 'license', 'repository', 'authors']}
        if package['name'] in ('fax', 'fax_derive'):
            target = destination / 'crates' / key / 'ATTRIBUTION.txt'
            target.write_text('Upstream declares MIT in Cargo.toml but supplies no separate license/copyright file.\n'
                              + 'Declared authors: ' + ', '.join(package['authors']) + '\n'
                              + 'The accompanying SPDX MIT text is a standard template, not an invented upstream copyright notice.\n'
                              + 'Exact unmodified source and original metadata are supplied in ../../sources/.\n')
            copied.append(str(target.relative_to(bundle)))
        entry['notices'] = copied
        # MPL source is shipped locally, not merely promised at a mutable URL.
        if 'MPL-2.0' in (package.get('license') or '') or package['name'] in ('fax', 'fax_derive'):
            target = destination / 'sources' / (key + '.tar.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            source_archive(directory, target, key)
            entry['sourceArchive'] = str(target.relative_to(bundle))
        entries.append(entry)
    shutil.copytree(LEGAL / 'onnxruntime-1.24.2', destination / 'onnxruntime-1.24.2')
    covered_sources = json.loads((LEGAL / 'sources.json').read_text())
    for record in covered_sources:
        source = ROOT / 'target/release-sources' / record['filename']
        if not source.is_file():
            raise SystemExit('Missing pinned covered source; run python3 scripts/fetch-release-sources.py')
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise SystemExit(f'Covered source checksum mismatch: {source}')
        (destination / 'sources').mkdir(exist_ok=True)
        shutil.copyfile(source, destination / 'sources' / record['filename'])
    shutil.copyfile(LEGAL / 'sources.json', destination / 'sources/upstream-sources.json')
    shutil.copyfile(LEGAL / 'provenance.json', destination / 'upstream-provenance.json')
    rust_docs = Path(output('rustc', '--print', 'sysroot')) / 'share/doc/rust'
    if not (rust_docs / 'licenses').is_dir() or not (rust_docs / 'COPYRIGHT-library.html').is_file():
        raise SystemExit('Missing Rust runtime attribution; install rust-docs for the active toolchain (rustup component add rust-docs) before packaging')
    rust_target = destination / 'rust-runtime'
    shutil.copytree(rust_docs / 'licenses', rust_target / 'licenses')
    shutil.copyfile(rust_docs / 'COPYRIGHT-library.html', rust_target / 'COPYRIGHT-library.html')
    shutil.copyfile(ROOT / 'vendor/lancedb/PATCHES.md', destination / 'LANCEDB-PATCHES.md')
    shutil.copyfile(ROOT / 'Cargo.lock', destination / 'Cargo.lock')
    inventory = {'scope': 'Resolved Linux workspace graph, including build/dev dependencies; conservative superset, not a claim every crate is linked.',
                 'target': host, 'cuda': args.cuda, 'rustc': output('rustc', '-V'),
                 'cargoLockSha256': hashlib.sha256((ROOT / 'Cargo.lock').read_bytes()).hexdigest(),
                 'packages': entries}
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    lines = ['# Third-party software', '', inventory['scope'], '',
             'Full upstream license and notice texts are in the linked directories. SPDX expressions are preserved; optional license alternatives are not relabeled as mandatory combined terms.', '',
             'ONNX Runtime 1.24.2 and its bundled dependencies: [license](onnxruntime-1.24.2/LICENSE), [notices](onnxruntime-1.24.2/ThirdPartyNotices.txt).',
             'Rust runtime: [copyright and third-party inventory](rust-runtime/COPYRIGHT-library.html), with full license texts alongside it. This upstream inventory also covers other targets (such as SGX/UEFI); their appearance does not mean they are linked into these Linux binaries.',
             'MPL-2.0 covered source is included in `sources/` for option-ext and the pinned Eigen revision used by ONNX Runtime. Eigen patches and build metadata are under `onnxruntime-1.24.2/`; apply both patches with `patch -p1` in the extracted Eigen directory. Source URLs and hashes are in `sources/upstream-sources.json`. Fractal Search is a separate MIT-licensed larger work.', '',
             '| Package | Declared license | Full texts |', '| --- | --- | --- |']
    for entry in entries:
        links = ', '.join(f'[{Path(p).name}]({p.removeprefix("third-party/")})' for p in entry['notices'])
        lines.append(f"| {entry['name']} {entry['version']} | {entry['license'] or 'See license file'} | {links} |")
    (destination / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Collected attribution for {len(entries)} resolved third-party packages')


if __name__ == '__main__':
    main()
