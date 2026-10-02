#!/usr/bin/env python3
"""Real MCP/stdin lifecycle checks against a temporary synthetic Quickshell catalog."""
import argparse
import base64
import hashlib
import json
import os
import pathlib
import queue
import shutil
import subprocess
import tempfile
import threading
from smoke_backend import Service, png


class Mcp:
    def __init__(self, binary, data, models, log):
        self.log = log.open("w")
        self.process = subprocess.Popen(
            [str(binary), "--data-dir", str(data), "--model-dir", str(models), "--provider", "cpu", "--threads", "2"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env={k: v for k, v in os.environ.items() if k != "FRACTAL_SEARCH_DEVICE_ID"},
        )
        self.messages = queue.Queue()
        self.next_id = 1
        self.history = []

        def read():
            try:
                for line in self.process.stdout:
                    assert len(line.encode()) <= 2 * 1024 * 1024 + 1
                    item = json.loads(line)
                    self.history.append(item)
                    self.messages.put(item)
            except Exception as error:
                self.messages.put(error)
            finally:
                self.messages.put(EOFError("MCP stdout closed"))
        threading.Thread(target=read, daemon=True).start()
        reply = self.call("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "fractal-mcp-smoke", "version": "1"}})
        assert reply["result"]["protocolVersion"] == "2025-11-25", reply
        self.notify("notifications/initialized")

    def write(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method):
        self.write({"jsonrpc": "2.0", "method": method})

    def send(self, method, params=None, **extras):
        ident = self.next_id
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": ident, "method": method, **extras}
        if params is not None:
            message["params"] = params
        self.write(message)
        return ident

    def receive(self, ident, timeout=120):
        while True:
            item = self.messages.get(timeout=timeout)
            if isinstance(item, Exception):
                raise item
            if item.get("id") == ident:
                return item
            raise AssertionError((ident, item))

    def call(self, method, params=None, **extras):
        return self.receive(self.send(method, params, **extras))

    def tool(self, name, arguments=None, success=True):
        reply = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        if success:
            assert "error" not in reply and not reply["result"].get("isError"), reply
        return reply

    def stop(self, expected=0):
        self.process.stdin.close()
        assert self.process.wait(timeout=120) == expected
        self.log.close()


def data(reply):
    return reply["result"]["structuredContent"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=pathlib.Path)
    parser.add_argument("mcp", type=pathlib.Path)
    parser.add_argument("models", type=pathlib.Path)
    args = parser.parse_args()
    backend, binary, models = args.backend.resolve(), args.mcp.resolve(), args.models.resolve()
    with tempfile.TemporaryDirectory(prefix="fractal-mcp-smoke-") as temporary:
        root = pathlib.Path(temporary)
        untouched = root / "providers-must-not-create"
        providers = json.loads(subprocess.check_output([str(binary), "--providers", "--data-dir", str(untouched), "--model-dir", str(untouched)], text=True))
        assert providers["providers"] == (["cpu", "cuda"] if providers["cudaCompiled"] else ["cpu"])
        assert not untouched.exists()
        photos, catalog = root / "private-originals", root / "data"
        photos.mkdir()
        png(photos / "red.png", 640, 480, (220, 20, 10))
        png(photos / "blue.png", 320, 200, (10, 20, 220))
        desktop = Service(backend, models, catalog, root / "desktop.log")
        try:
            indexed = desktop.call({"type": "index", "directory": str(photos)})
            assert indexed["indexed"] == 2, indexed
            # No agent operation may depend on originals, even for thumbnails.
            shutil.rmtree(photos)
            original_hash = hashlib.sha256((catalog / "search.db").read_bytes()).hexdigest()
            mcp = Mcp(binary, catalog, root / "missing-models", root / "metadata.log")
            tools = mcp.call("tools/list")["result"]["tools"]
            assert {tool["name"] for tool in tools} == {"status", "browse_assets", "search_assets", "get_asset_metadata", "get_thumbnail"}
            assert all(tool["inputSchema"]["additionalProperties"] is False and tool["annotations"]["readOnlyHint"] for tool in tools)
            assert data(mcp.tool("status"))["textModelLoaded"] is False
            page = data(mcp.tool("browse_assets", {"limit": 1}))
            assert page["total"] == 2 and len(page["items"]) == 1
            asset = page["items"][0]
            second = data(mcp.tool("browse_assets", {"offset": 1, "limit": 1}))["items"][0]
            assert asset["assetId"] != second["assetId"]
            assert data(mcp.tool("get_asset_metadata", {"assetId": asset["assetId"]})) == asset
            thumbnail = mcp.tool("get_thumbnail", {"thumbnailHandle": asset["thumbnailHandle"]})["result"]["content"][0]
            assert thumbnail["type"] == "image" and thumbnail["mimeType"] == "image/webp"
            raw = base64.b64decode(thumbnail["data"], validate=True)
            assert raw.startswith(b"RIFF") and len(raw) <= 1024 * 1024
            for tool, arguments in [("status", {"path": str(photos)}), ("browse_assets", {"limit": 101}), ("search_assets", {"query": ""}), ("search_assets", {"query": "é" * 3000}), ("get_asset_metadata", {"assetId": str(photos)}), ("get_thumbnail", {"thumbnailHandle": "../../private-originals/red.png"})]:
                assert "error" in mcp.tool(tool, arguments, success=False)
            assert "error" in mcp.call("tools/list", {}, unexpected=True)
            assert "error" in mcp.call("tools/call", {"name": "status", "arguments": {}, "extra": True})
            mcp.process.stdin.write("invalid json\n")
            mcp.process.stdin.flush()
            assert mcp.receive(None)["error"]["code"] == -32700
            failed = mcp.tool("search_assets", {"query": "red picture"}, success=False)
            assert failed["result"]["isError"]
            assert data(mcp.tool("status"))["textModelLoaded"] is False
            assert not any(str(root) in json.dumps(message) or "file://" in json.dumps(message) for message in mcp.history)
            mcp.stop()
            assert "Loaded text.onnx" not in (root / "metadata.log").read_text()
            assert hashlib.sha256((catalog / "search.db").read_bytes()).hexdigest() == original_hash
            # Server and desktop remain concurrent. Model loads only for search.
            mcp = Mcp(binary, catalog, models, root / "search.log")
            assert data(mcp.tool("status"))["textModelLoaded"] is False
            hits = data(mcp.tool("search_assets", {"query": "a red picture", "limit": 2}))["items"]
            assert len(hits) == 2 and hits[0]["similarity"] >= hits[1]["similarity"]
            assert data(mcp.tool("status"))["textModelLoaded"] is True
            # Send multiple accepted requests then EOF: every response must flush.
            ids = [mcp.send("tools/call", {"name": "search_assets", "arguments": {"query": "blue picture", "limit": 1}}) for _ in range(3)]
            mcp.process.stdin.close()
            for ident in ids:
                assert len(data(mcp.receive(ident))["items"]) == 1
            assert mcp.process.wait(timeout=120) == 0
            mcp.log.close()
            assert "Loaded text.onnx" in (root / "search.log").read_text()
            assert "Loaded visual.onnx" not in (root / "search.log").read_text()
            assert not any(str(root) in json.dumps(message) or "file://" in json.dumps(message) for message in mcp.history)
            # Invalid frames terminate explicitly after a bounded error response.
            mcp = Mcp(binary, catalog, models, root / "overlong.log")
            mcp.process.stdin.write("x" * 65537 + "\n")
            mcp.process.stdin.flush()
            assert mcp.receive(None)["error"]["code"] == -32600
            mcp.stop(expected=1)
            assert desktop.call({"type": "status"})["catalog"]["imageCount"] == 2
            desktop.call({"type": "shutdown"})
            assert desktop.process.wait(timeout=60) == 0
        finally:
            if desktop.process.poll() is None:
                desktop.process.terminate()
                desktop.process.wait(timeout=60)
            desktop.log.close()
    print("MCP smoke passed: strict lifecycle/tools, read-only concurrent catalog, hidden originals, bounded sanitized thumbnails, lazy real text inference, EOF drain and frame rejection")


if __name__ == "__main__":
    main()
