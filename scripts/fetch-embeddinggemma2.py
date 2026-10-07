#!/usr/bin/env python3
"""Fetch pinned EmbeddingGemma 2 text/image/audio/video assets outside the repo and verify them.

This is explicit model setup, never an automatic validation dependency. The shared
vision encoder handles images and video; the audio encoder handles audio.
"""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise RuntimeError(f"{path}: expected {spec['bytes']} bytes")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{path}: SHA-256 mismatch: {digest}")


class HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise RuntimeError("Refusing a non-HTTPS model download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.is_relative_to(repo_root):
        raise RuntimeError("Model assets must be stored outside the repository")
    manifest = json.loads((repo_root / "crates/fractal-search-core/models/embeddinggemma2.json").read_text())
    opener = urllib.request.build_opener(HttpsRedirectHandler())
    for spec in manifest["files"]:
        path = destination / spec["path"]
        if not path.resolve().is_relative_to(destination):
            raise RuntimeError(f"Model asset escapes destination: {path}")
        if path.exists():
            verify(path, spec)
            print(f"Verified existing {path}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {spec['path']} ({spec['bytes']} bytes)", flush=True)
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{spec['path']}"
        if partial.is_symlink():
            raise RuntimeError(f"Refusing a symlink download destination: {partial}")
        received = 0
        with opener.open(url, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > spec["bytes"]:
                    raise RuntimeError(f"Download exceeds expected size: {spec['path']}")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify(partial, spec)
        partial.replace(path)
        print(f"Verified {path}", flush=True)


if __name__ == "__main__":
    main()
