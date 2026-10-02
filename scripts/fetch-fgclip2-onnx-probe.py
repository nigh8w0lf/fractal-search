#!/usr/bin/env python3
"""Fetch pinned developer probe assets outside the repo; verify size and SHA-256."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise RuntimeError(f"{path}: expected {spec['bytes']} bytes")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{path}: SHA-256 mismatch: {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.is_relative_to(repo_root):
        raise RuntimeError("Probe model assets must be stored outside the repository")
    manifest = json.loads((repo_root / "crates/fractal-search-core/models/fgclip2.json").read_text())
    for spec in manifest["files"]:
        path = destination / spec["path"]
        if path.exists():
            verify(path, spec)
            print(f"Verified existing {path}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {spec['path']} ({spec['bytes']} bytes)", flush=True)
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{spec['path']}"
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        verify(partial, spec)
        partial.replace(path)
        print(f"Verified {path}", flush=True)


if __name__ == "__main__":
    main()
