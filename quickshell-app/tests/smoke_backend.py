#!/usr/bin/env python3
"""Opt-in real-inference transport/lifecycle checks; only uses temporary catalogs."""

import argparse
import json
import math
import pathlib
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import zlib


def png(path, width=256, height=192, color=(180, 60, 40)):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    pixels = (b"\0" + bytes(color) * width) * height
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def local_url(value):
    url = urllib.parse.urlsplit(value)
    assert url.scheme == "file" and not url.netloc
    return pathlib.Path(urllib.parse.unquote(url.path))


class Service:
    def __init__(self, binary, models, data, log):
        self.args = [
            str(binary), "--model-dir", str(models), "--data-dir", str(data), "serve"
        ]
        self.log = log.open("w")
        self.process = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.messages = queue.Queue()
        self.replies = {}
        self.next_id = 1
        self.shutting_down = False

        def read():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except Exception as error:
                self.messages.put(error)
            finally:
                self.messages.put(EOFError("Search service stdout closed"))

        threading.Thread(target=read, daemon=True).start()

    def send(self, command):
        if command["type"] == "shutdown":
            self.shutting_down = True
        request_id = self.next_id
        self.next_id += 1
        self.process.stdin.write(json.dumps({"id": request_id, "command": command}) + "\n")
        self.process.stdin.flush()
        return request_id

    def until(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while True:
            message = self.messages.get(timeout=max(.01, deadline - time.monotonic()))
            if isinstance(message, Exception):
                raise message
            if "ok" in message:
                self.replies[message["id"]] = message
            if predicate(message):
                return message
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the search service")

    def response(self, request_id, success=True):
        message = self.replies.pop(request_id, None)
        if message is None:
            message = self.until(lambda item: item.get("id") == request_id and "ok" in item)
            self.replies.pop(request_id, None)
        assert message["ok"] is success, message
        return message["result"] if success else message["error"]

    def wait_for_completed_image(self, request_id):
        self.until(
            lambda item: item.get("id") == request_id
            and item.get("event") == "indexProgress"
            and item["payload"]["current"] >= 1
        )

    def call(self, command, success=True):
        return self.response(self.send(command), success)

    def search_with_progress(self, command):
        request_id = self.send(command)
        stages = []

        def completed(message):
            if message.get("id") != request_id:
                return False
            if message.get("event") == "searchProgress":
                stages.append(message["payload"]["stage"])
            return "ok" in message

        self.until(completed)
        return self.response(request_id), stages

    def close(self):
        if self.process.poll() is None and not self.shutting_down:
            self.call({"type": "shutdown"})
        assert self.process.wait(timeout=15) == 0
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("models", type=pathlib.Path)
    parser.add_argument(
        "--fixtures", type=pathlib.Path,
        help="Optional native-fixtures directory for JPEG/WebP coverage",
    )
    args = parser.parse_args()
    binary, models = args.binary.resolve(strict=True), args.models.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="fractal-search-smoke-", dir="/tmp") as temporary:
        root = pathlib.Path(temporary)
        photos = root / "photos"
        photos.mkdir()
        png(photos / "red # ü.png")
        png(photos / "blue.png", color=(20, 60, 210))
        if args.fixtures:
            for name in ["decode-jpeg.jpg", "decode-webp.webp", "decode-webp-misnamed.jpg"]:
                shutil.copyfile(args.fixtures / name, photos / name)
        count = len(list(photos.iterdir()))
        # Keep a genuinely cold text-load check separate from image indexing:
        # EmbeddingGemma's image path also loads the shared text backbone.
        cold = Service(binary, models, root / "cold-data", root / "cold.log")
        try:
            ready = cold.until(lambda item: item.get("event") == "ready")["payload"]
            assert ready["modelLoaded"] is False and ready["textModelLoaded"] is False
            empty, stages = cold.search_with_progress({"type": "search", "query": "a cold-start description", "limit": 2})
            assert stages == ["starting", "searching"], stages
            assert empty["items"] == [] and empty["total"] == 0
            assert cold.call({"type": "status"})["textModelLoaded"] is True
            print("Cold text-model startup progress passed", flush=True)
        finally:
            cold.close()
        service = Service(binary, models, root / "data", root / "backend.log")
        try:
            ready = service.until(lambda item: item.get("event") == "ready")["payload"]
            assert ready["catalog"]["assetCount"] == 0 and ready["modelLoaded"] is False
            assert ready["catalog"]["modelProfile"] == "embeddinggemma2-daa72c51-fp32-multimodal-mrl512-v2"
            assert ready["textModelLoaded"] is False
            preferences = {"topMatches": 6, "resultLayout": "listPreview", "watchChanges": False, "closeAfterOpening": True, "imageBatchSize": 1}
            assert ready["settings"] == preferences
            assert "Top matches" in service.call({"type": "updateSettings", "settings": {**preferences, "topMatches": 0}}, success=False)
            preferences.update(topMatches=20, resultLayout="list", closeAfterOpening=False)
            assert service.call({"type": "updateSettings", "settings": preferences})["settings"] == preferences
            assert service.call({"type": "browse"})["items"] == []
            locked = subprocess.run(
                [str(binary), "--data-dir", str(root / "data"), "status"],
                capture_output=True, text=True, timeout=10,
            )
            assert locked.returncode != 0 and "already open" in locked.stderr
            assert "limit" in service.call({"type": "browse", "limit": 0}, success=False)
            service.send({"type": "status", "unknown": True})
            rejection = service.until(
                lambda item: item.get("id") is None and item.get("ok") is False
            )
            assert "unknown field" in rejection["error"]
            # Model warmup plus several FP32 image embeddings is a long-running
            # operation on CPU, unlike the metadata calls' 60-second deadline.
            index_id = service.send({"type": "index", "directory": photos.as_uri()})
            service.until(lambda item: item.get("id") == index_id and "ok" in item, timeout=300)
            result = service.response(index_id)
            assert result["indexed"] == count and not result["cancelled"]
            assets = service.call({"type": "browse"})["items"]
            identifiers = {item["assetId"] for item in assets}
            assert len(identifiers) == count
            for item in assets:
                assert item["assetId"].startswith("asset_") and local_url(item["thumbnailUrl"]).is_file()
            print("Index, file URLs and thumbnail generation passed", flush=True)
            preview_asset = next(item for item in assets if item["filename"] == "red # ü.png")
            preview = service.call({"type": "preview", "assetId": preview_asset["assetId"]})
            preview_path = local_url(preview["url"])
            assert preview_path.is_file()
            assert struct.unpack(">II", preview_path.read_bytes()[16:24]) == (256, 192)
            assert local_url(service.call({"type": "originalUrl", "assetId": assets[0]["assetId"]})["url"]).is_file()
            # Image indexing has already loaded the shared backbone; search is warm.
            assert service.call({"type": "status"})["textModelLoaded"] is True
            hits, stages = service.search_with_progress({"type": "search", "query": "a colorful image", "limit": 2})
            assert stages == ["searching"], stages
            assert len(hits["items"]) == 2 and hits["total"] == count
            assert all(math.isfinite(item["similarity"]) for item in hits["items"])
            page, stages = service.search_with_progress({"type": "search", "query": "a colorful image", "offset": 2, "limit": 2})
            assert stages == [], stages  # Cached pagination needs no model work.
            assert not ({item["assetId"] for item in hits["items"]} & {item["assetId"] for item in page["items"]})
            _, stages = service.search_with_progress({"type": "search", "query": "a blue image", "limit": 2})
            assert stages == ["searching"], stages
            print("Image-warmed shared backbone, warm search and cached-page progress passed", flush=True)
            assert service.call({"type": "index", "directory": str(photos)})["unchanged"] == count
            local_url(assets[0]["thumbnailUrl"]).unlink()
            assert "Thumbnail is unavailable" in service.call({"type": "browse"}, success=False)
            assert service.call({"type": "index", "directory": str(photos)})["indexed"] == 1
            assert identifiers == {item["assetId"] for item in service.call({"type": "browse"})["items"]}
            corrupt = root / "corrupt"
            corrupt.mkdir()
            png(corrupt / "a-valid.png")
            (corrupt / "b-bad.webp").write_bytes(b"not an image")
            png(corrupt / "d-valid.png", color=(20, 60, 210))
            jpeg_fixture = pathlib.Path(__file__).resolve().parents[2] / "crates/fractal-search-core/tests/fixtures/jpeg-extraneous-byte.jpg.hex"
            (corrupt / "c-corrupt.jpg").write_bytes(bytes.fromhex(jpeg_fixture.read_text(encoding="ascii")))
            expected_failures = 2
            index_id = service.send({"type": "index", "directory": str(corrupt)})
            failure_event = service.until(
                lambda item: item.get("id") == index_id
                and item.get("event") == "indexProgress"
                and "failure" in item["payload"]
            )["payload"]
            assert failure_event["current"] == 2 and failure_event["failed"] == 1
            assert failure_event["failure"]["path"].endswith("b-bad.webp")
            assert failure_event["failure"]["error"]
            result = service.response(index_id)
            assert result["indexed"] == 2 and result["failed"] == expected_failures
            assert not result["cancelled"] and result["failuresOmitted"] == 0
            assert len(result["failures"]) == expected_failures
            assert "1 extraneous bytes before marker 0xd9" in result["failures"][1]["error"]
            assert service.call({"type": "browse"})["total"] == count + 2
            png(corrupt / "b-bad.webp")  # Content, rather than its suffix, selects the decoder.
            if args.fixtures:
                shutil.copyfile(args.fixtures / "decode-jpeg.jpg", corrupt / "c-corrupt.jpg")
            else:
                png(corrupt / "c-corrupt.jpg")
            result = service.call({"type": "index", "directory": str(corrupt)})
            assert result["indexed"] == expected_failures and result["unchanged"] == 2
            assert result["failed"] == 0 and result["failures"] == []
            additional_count = 2 + expected_failures
            assert service.call({"type": "index", "directory": str(photos)})["unchanged"] == count
            sources = service.call({"type": "status"})["catalog"]["sources"]
            photo_source = next(source["id"] for source in sources if source["directory"] == str(photos))
            corrupt_source = next(source["id"] for source in sources if source["directory"] == str(corrupt))
            filtered = service.call({"type": "browse", "sourceId": photo_source})
            assert filtered["total"] == count
            assert identifiers == {item["assetId"] for item in filtered["items"]}
            filtered = service.call({"type": "search", "query": "a colorful image", "sourceId": corrupt_source, "maxResults": 1})
            assert filtered["total"] == 1 and pathlib.Path(filtered["items"][0]["path"]).parent == corrupt
            assert service.call({"type": "search", "query": "a colorful image", "sourceId": corrupt_source, "maxResults": 2})["total"] == 2
            assert "maxResults" in service.call({"type": "search", "query": "red", "maxResults": 0}, success=False)
            assert "does not exist" in service.call({"type": "browse", "sourceId": 999999}, success=False)
            print("Search pagination, preview, rescan, corrupt-file continuation and repair passed", flush=True)

            cancel_photos = root / "cancel"
            cancel_photos.mkdir()
            for i in range(8):
                png(cancel_photos / f"{i}.png", 512, 512)
            index_id = service.send({"type": "index", "directory": str(cancel_photos)})
            service.wait_for_completed_image(index_id)
            second_id = service.send({"type": "index", "directory": str(photos)})
            status_id = service.send({"type": "status"})
            cancel_id = service.send({"type": "cancel"})
            assert "already active" in service.response(second_id, success=False)
            assert service.response(cancel_id)["cancelRequested"]
            result = service.response(index_id)
            assert result["cancelled"] and 1 <= result["indexed"] < 8
            assert service.response(status_id)["catalog"]["assetCount"] == count + additional_count + result["indexed"]
            assert not service.call({"type": "index", "directory": str(photos)})["cancelled"]
            print("Cancellation finishes active work and rejects concurrent scans", flush=True)

            before_removal = service.call({"type": "browse", "sourceId": corrupt_source})["items"]
            assert service.call({"type": "removeSource", "sourceId": corrupt_source})["removed"] == additional_count
            assert all(pathlib.Path(item["path"]).is_file() for item in before_removal)
            assert all(not local_url(item["thumbnailUrl"]).exists() for item in before_removal)
            assert "does not exist" in service.call({"type": "removeSource", "sourceId": corrupt_source}, success=False)

            watched = root / "watched"
            watched.mkdir()
            assert service.call({"type": "index", "directory": str(watched)})["indexed"] == 0
            watched_source = next(source["id"] for source in service.call({"type": "status"})["catalog"]["sources"] if source["directory"] == str(watched))
            preferences["watchChanges"] = True
            assert service.call({"type": "updateSettings", "settings": preferences})["settings"] == preferences
            png(watched / "first.png")
            started = service.until(lambda item: item.get("event") == "indexStarted")
            assert started["payload"]["directory"] == str(watched)
            progress = service.until(lambda item: item.get("event") == "indexProgress" and item.get("id") is None)
            assert progress["payload"]["directory"] == str(watched)
            result = service.until(lambda item: item.get("event") == "indexFinished")["payload"]
            assert result["indexed"] == 1 and not result["cancelled"] and result["failed"] == 0
            # Coalesce replacement + creation into one complete rescan.
            png(watched / "first.png", color=(50, 160, 80))
            png(watched / "second.png", color=(20, 60, 210))
            result = service.until(lambda item: item.get("event") == "indexFinished")["payload"]
            assert result["indexed"] == 2
            assert service.call({"type": "browse", "sourceId": watched_source})["total"] == 2
            (watched / "first.png").unlink()
            result = service.until(lambda item: item.get("event") == "indexFinished")["payload"]
            assert result["removed"] == 1 and result["unchanged"] == 1
            assert service.call({"type": "removeSource", "sourceId": watched_source})["removed"] == 1
            preferences["watchChanges"] = False
            assert service.call({"type": "updateSettings", "settings": preferences})["settings"] == preferences
            assert (watched / "second.png").is_file()
            print("Saved preferences, filtered top-k, source removal and native change watching passed", flush=True)

            closing = root / "closing"
            closing.mkdir()
            for i in range(6):
                png(closing / f"{i}.png", 512, 512, (50, 160, 80))
            index_id = service.send({"type": "index", "directory": str(closing)})
            service.wait_for_completed_image(index_id)
            shutdown_id = service.send({"type": "shutdown"})
            result = service.response(index_id)
            assert result["cancelled"] and 1 <= result["indexed"] < 6
            assert service.response(shutdown_id)["stopped"]
            service.close()
            status = subprocess.run(
                [str(binary), "--data-dir", str(root / "data"), "status"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            assert json.loads(status.stdout)["catalog"]["assetCount"] > count
            assert json.loads(status.stdout)["settings"] == preferences
            eof = subprocess.run(
                [str(binary), "--data-dir", str(root / "data"), "serve"],
                input='{"id":1,"command":{"type":"status"}}\n'
                '{"id":2,"command":{"type":"browse","limit":1}}\n',
                capture_output=True, text=True, check=True, timeout=15,
            )
            messages = [json.loads(line) for line in eof.stdout.splitlines()]
            assert {item["id"] for item in messages if item.get("ok")} == {1, 2}
            assert "drained and stopped" in eof.stderr
            print("Shutdown persists completed images; reopen and EOF drain passed", flush=True)

            # Exercise the actual native signal handler during inference as well
            # as the UI's explicit shutdown request above.
            signalled = Service(binary, models, root / "signal-data", root / "signal.log")
            try:
                signalled.until(lambda item: item.get("event") == "ready")
                index_id = signalled.send({"type": "index", "directory": str(closing)})
                signalled.wait_for_completed_image(index_id)
                signalled.process.terminate()
                assert signalled.response(index_id)["cancelled"]
                assert signalled.process.wait(timeout=15) == 0
                print("SIGTERM drains active inference and exits normally", flush=True)
            finally:
                signalled.close()
        finally:
            if service.process.poll() is None:
                service.close()


if __name__ == "__main__":
    main()
