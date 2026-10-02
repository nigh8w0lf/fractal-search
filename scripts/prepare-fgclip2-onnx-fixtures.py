#!/usr/bin/env python3
"""Offline dynamic-patch fixtures for the zihuv ONNX export, not a runtime provider.

Use an isolated development venv as described in docs/MODELS.md. Models must
already be cached; no downloads or catalog writes.
The Rust probe calculates position embeddings and looks up text tokens itself.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer


UPSTREAM_REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"


def measure(call):
    with torch.inference_mode():
        call()
        times = []
        for _ in range(5):
            start = time.perf_counter()
            output = call()
            output = output / output.norm(p=2, dim=-1, keepdim=True)
            times.append((time.perf_counter() - start) * 1000)
    return output.tolist(), statistics.median(times)


def image(width, height):
    yy, xx = np.indices((height, width))
    pixels = np.stack([(xx * 3 + yy) % 256, (xx + yy * 2) % 256,
                       ((xx // 64 + yy // 64) % 2) * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


def patch_budget(source):
    count = (source.width // 16) * (source.height // 16)
    for threshold, budget in [(784, 1024), (576, 784), (256, 576), (128, 256)]:
        if count > threshold:
            return budget
    return 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    pillow_heif.register_heif_opener()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    model_id = "qihoo360/fg-clip2-base"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=UPSTREAM_REVISION, trust_remote_code=True,
        local_files_only=True,
    ).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=UPSTREAM_REVISION, local_files_only=True,
    )
    processor = AutoImageProcessor.from_pretrained(
        model_id, revision=UPSTREAM_REVISION, local_files_only=True,
    )
    exported_tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    exported_tokenizer.enable_truncation(max_length=64)
    exported_tokenizer.enable_padding(length=64, pad_id=0, pad_token="<pad>")
    positions = np.fromfile(args.model_dir / "vision_pos_embedding.bin", dtype="<f4").reshape(16, 16, 768)
    original_positions = model.vision_model.embeddings.position_embedding.weight.detach().numpy().reshape(16, 16, 768)
    if not np.array_equal(positions, original_positions):
        raise RuntimeError("Exported vision position table differs from pinned upstream model")
    token_table = np.memmap(args.model_dir / "text_token_embedding.bin", dtype="<f2", mode="r", shape=(256000, 768))
    fixtures = []

    def write_tensor(name, suffix, array):
        filename = name + "-" + suffix + ".bin"
        array.tofile(args.output_dir / filename)
        return filename

    texts = ["a red dress outdoors", "海边的日落", "Mixed CASE café portrait",
             "a photo of " + "a colorful tree beside a lake " * 30,
             "", "  multiple   spaces\tand\nlines  ", "ΟΣ İSTANBUL STRAẞE",
             "👩🏽‍💻 café e\u0301 東京 العربية", "<bos> literal <eos> <pad>"]
    for i, text in enumerate(texts):
        name = f"text-{i}"
        inputs = tokenizer([text.lower()], padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt")
        ids = inputs["input_ids"]
        if exported_tokenizer.encode(text.lower()).ids != ids[0].tolist():
            raise RuntimeError(f"Tokenizer mismatch for fixture {i}")
        original_tokens = model.text_model.embeddings.token_embedding(ids).detach().numpy().astype(np.float16)
        if not np.array_equal(token_table[ids.numpy()], original_tokens):
            raise RuntimeError(f"Split fp16 token table mismatch for fixture {i}")
        reference, median = measure(lambda: model.get_text_features(**inputs, walk_type="short"))
        fixtures.append(dict(name=name, kind="text", text=text,
                             input_file=write_tensor(name, "ids", ids.numpy().astype("<i4")),
                             shape=list(ids.shape), reference=reference, reference_median_ms=median))
        print(name, "reference_ms=", round(median, 3), flush=True)

    sizes = {128: [(128, 128), (96, 192), (192, 96)],
             256: [(224, 224), (160, 320), (320, 160)],
             576: [(320, 320), (240, 480), (480, 240)],
             784: [(400, 400), (288, 576), (576, 288)],
             1024: [(512, 512), (736, 1181), (1600, 900)]}
    cases = [(f"image-{w}x{h}-p{budget}", [image(w, h)], budget)
             for budget, shapes in sizes.items() for w, h in shapes]
    cases += [("image-batch2-p1024", [image(736, 1181), image(1600, 900)], 1024),
              ("image-batch8-p128", [image(*sizes[128][i % 3]) for i in range(8)], 128)]
    # Store encoded source files so the Rust probe can exercise native decoding.
    sources = {}
    for name, images, _ in cases:
        files = []
        for i, source in enumerate(images):
            filename = f"{name}-source-{i}.png"
            source.save(args.output_dir / filename)
            files.append(filename)
        sources[name] = files
    for label, source, extension, options in [
        ("jpeg", image(736, 1181), "jpg", {"quality": 91}),
        ("jpeg-gray", image(320, 160).convert("L"), "jpg", {}),
        ("webp", image(736, 1181), "webp", {"quality": 85}),
        ("webp-misnamed", image(736, 1181), "jpg", {"format": "WEBP", "quality": 85}),
        ("png-alpha", image(241, 483).convert("RGBA"), "png", {}),
        ("png-gray", image(99, 193).convert("L"), "png", {}),
        ("heic", image(320, 240), "heic", {"format": "HEIF", "quality": 85}),
        ("tiny", image(1, 1), "png", {}),
        ("thin", image(7, 1537), "png", {}),
    ]:
        name = "decode-" + label
        filename = f"{name}.{extension}"
        source.save(args.output_dir / filename, **options)
        with Image.open(args.output_dir / filename) as encoded:
            decoded = encoded.convert("RGB")
            decoded.load()
        cases.append((name, [decoded], patch_budget(decoded)))
        sources[name] = [filename]
    for name, images, budget in cases:
        if max(map(patch_budget, images)) != budget:
            raise RuntimeError(f"Fixture {name} does not match the pinned dynamic patch-budget policy")
        inputs = processor(images=images, max_num_patches=budget, return_tensors="pt")
        reference, median = measure(lambda: model.get_image_features(**inputs))
        spatial = inputs["spatial_shapes"]
        resized_positions = model.vision_model.embeddings.resize_positional_embeddings(
            torch.from_numpy(positions.copy()), spatial, max_length=budget)
        fixtures.append(dict(name=name, kind="image", image_files=sources[name],
                             input_file=write_tensor(name, "pixels", inputs["pixel_values"].numpy().astype("<f4")),
                             shape=list(inputs["pixel_values"].shape),
                             mask_file=write_tensor(name, "mask", inputs["pixel_attention_mask"].numpy().astype("<i4")),
                             spatial_shapes=spatial.tolist(),
                             position_reference_file=write_tensor(name, "positions", resized_positions.numpy().astype("<f4")),
                             reference=reference, reference_median_ms=median))
        print(name, "spatial_shapes=", spatial.tolist(), "reference_ms=", round(median, 3), flush=True)

    (args.output_dir / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    (args.output_dir / "environment.json").write_text(json.dumps(dict(
        upstream_revision=UPSTREAM_REVISION, torch=torch.__version__,
        intra_threads=6, interop_threads=1, image_processor=type(processor).__name__,
        attention_implementation=model.config._attn_implementation,
        note="Synthetic images; dynamic patch budgets; preprocessing and IPC excluded from inference timing",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
