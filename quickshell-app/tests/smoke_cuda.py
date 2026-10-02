#!/usr/bin/env python3
"""Explicit CUDA selection and no-retry checks using synthetic temporary catalogs.

Requires the CUDA-enabled backend, pinned models, NVIDIA driver and CUDA libraries.
Run validate_cuda separately for embedding parity and measured GPU graph placement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from smoke_backend import png
from smoke_plugin import Client, stop


def launch(binary, models, root, device_id, env, warm=False):
    root.mkdir()
    socket = root / "run/backend.sock"
    log = (root / "daemon.log").open("w")
    command = [str(binary), "daemon", "--socket", str(socket), "--data-dir", str(root / "data"),
               "--model-dir", str(models), "--provider", "cuda", "--device-id", str(device_id)]
    if warm:
        command.append("--warm-search")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, env=env)
    deadline = time.monotonic() + 15
    while not socket.exists():
        assert process.poll() is None, (root / "daemon.log").read_text()
        assert time.monotonic() < deadline, "Daemon socket did not appear"
        time.sleep(.03)
    client = Client(socket)
    ready = client.until(lambda value: value.get("event") == "ready")["payload"]
    assert ready["provider"] == "CUDA" and ready["deviceId"] == device_id, ready
    assert not ready["modelLoaded"], ready
    return process, client, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    binary, models = args.binary.resolve(strict=True), args.models.resolve(strict=True)
    env = os.environ.copy()
    env["FRACTAL_SEARCH_PROVIDER"] = "cpu"  # Explicit daemon flag must win and reach worker.
    env.pop("FRACTAL_SEARCH_DEVICE_ID", None)
    with tempfile.TemporaryDirectory(prefix="fractal-search-cuda-smoke-", dir="/tmp") as temp:
        root = Path(temp)
        absent = root / "must-not-create"
        result = subprocess.run([str(binary), "providers", "--data-dir", str(absent)],
                                env=env, check=True, capture_output=True, text=True)
        assert json.loads(result.stdout) == {"cudaCompiled": True, "providers": ["cpu", "cuda"]}
        assert not absent.exists()
        for options in [["--provider", "cpu", "--device-id", "0"],
                        ["--provider", "cuda", "--device-id=-1"]]:
            result = subprocess.run([str(binary), "status", "--data-dir", str(absent), *options],
                                    env=env, capture_output=True, text=True)
            assert result.returncode != 0 and "[Fatal]" in result.stderr, result
            assert not absent.exists()

        photos = root / "photos"
        photos.mkdir()
        png(photos / "red.png", color=(230, 30, 10))
        png(photos / "blue.png", color=(10, 40, 230))
        process, client, log = launch(binary, models, root / "valid", args.device_id, env)
        try:
            client.send(1, {"type": "setModelDirectory", "directory": str(models)})
            updated = client.response(1)["result"]
            assert updated["provider"] == "CUDA" and updated["deviceId"] == args.device_id
            client.send(2, {"type": "index", "directory": str(photos)})
            assert client.response(2)["result"]["indexed"] == 2
            client.send(3, {"type": "search", "query": "red image", "limit": 2, "maxResults": 2})
            assert client.response(3)["result"]["total"] == 2
            client.send(4, {"type": "status"})
            status = client.response(4)["result"]
            assert status["provider"] == "CUDA" and status["modelLoaded"] and status["textModelLoaded"]
            client.send(5, {"type": "shutdown"})
            client.response(5)
            assert process.wait(timeout=30) == 0
        finally:
            stop(process)
            client.close()
            log.close()

        process, client, log = launch(binary, models, root / "invalid", 2_147_483_647, env, warm=True)
        try:
            finished = client.until(lambda value: value.get("event") == "warmupFinished")["payload"]
            assert finished["ok"] is False and "CUDA" in finished["error"], finished
            client.send(1, {"type": "status"})
            status = client.response(1)["result"]
            assert status["provider"] == "CUDA" and not status["modelLoaded"], status
            client.send(2, {"type": "shutdown"})
            client.response(2)
            assert process.wait(timeout=30) == 0
        finally:
            stop(process)
            client.close()
            log.close()
        print("CUDA daemon forwarding, real indexing/search, model path retention, invalid-device failure and clean shutdown passed")


if __name__ == "__main__":
    main()
