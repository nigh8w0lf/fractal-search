#!/usr/bin/env python3
"""Create offline synthetic EmbeddingGemma 2 preprocessing and ONNX parity fixtures.

Requires NumPy, Pillow, PyTorch, Torchvision and Transformers with
Gemma4ImageProcessor support. Full parity also needs ONNX Runtime and an
existing verified FP32 model cache. --preprocess-only uses only the verified
config/tokenizer files and never opens a model graph or weights.
No network access, remote model code, catalog access, or weight downloads occur.
The default reference projects native output to leading-512/L2-normalized indexed
vectors. --native-output preserves full 768-dimensional graph-parity references.
The independent reference is Hugging Face's image processor/tokenizer plus the
pinned ONNX export, not a claim of parity against Google's original checkpoint.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from embeddinggemma2_contract import (
    EMBEDDING_DIM, MODEL_PROFILE, NATIVE_EMBEDDING_DIM, NATIVE_MODEL_PROFILE,
    project_indexed_vector,
)


QUERY_PREFIX = "task: search result | query: "
MAX_TEXT_TOKENS = 8192
REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"


def verify_cache(directory, preprocess_only=False):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "crates/fractal-search-core/models/embeddinggemma2.json").read_text())
    if manifest["revision"] != REVISION:
        raise RuntimeError("Update the fixture reference when changing the model revision")
    for spec in manifest["files"]:
        if preprocess_only and spec["path"].startswith("onnx/"):
            continue
        path = directory / spec["path"]
        if path.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"Invalid cached model asset size: {path}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"Invalid cached model asset SHA-256: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--native-output", action="store_true",
                        help="Keep full 768-dimensional native graph outputs for explicit parity validation")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Generate processor/tokenizer tensors without model weights or inference")
    parser.add_argument("--max-image-batch", type=int, choices=range(1, 9), default=8,
                        help="Explicitly bound image batch fixtures for available RAM (default: 8)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Reference inference repeats per fixture (default: 1)")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    repo_root = Path(__file__).resolve().parents[1]
    models = args.model_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(repo_root) or output.is_relative_to(models):
        raise RuntimeError("Fixtures must be written outside the source and model directories")
    if output.exists():
        raise RuntimeError("Use a new output directory to protect existing fixtures")
    manifest = verify_cache(models, args.preprocess_only)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import PIL
    import torch
    import torchvision
    import transformers
    from PIL import Image
    from transformers import AutoTokenizer, Gemma4ImageProcessor, Gemma4AudioFeatureExtractor

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    processor_config = json.loads((models / "processor_config.json").read_text())
    processor = Gemma4ImageProcessor(**processor_config["image_processor"])
    video_processor = Gemma4ImageProcessor(**{
        **processor_config["image_processor"],
        "max_soft_tokens": processor_config["video_processor"]["max_soft_tokens"],
    })
    audio_processor = Gemma4AudioFeatureExtractor(**processor_config["feature_extractor"])
    tokenizer = AutoTokenizer.from_pretrained(models, local_files_only=True, trust_remote_code=False)
    ort = None
    if not args.preprocess_only:
        import onnxruntime as ort

    options = ort.SessionOptions() if ort else None
    if not args.preprocess_only:
        options.intra_op_num_threads = 6
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(str(models / "onnx/model.onnx"), options,
                                    providers=["CPUExecutionProvider"])
    vision = None
    audio_encoder = None
    output.mkdir(parents=True)
    empty = np.empty((0, 512), dtype=np.float32)
    fixtures = []

    def write_tensor(name, suffix, array):
        filename = f"{name}-{suffix}.bin"
        array.tofile(output / filename)
        return filename

    def tokens(text, plain_text=False):
        result = tokenizer(text, padding=True, truncation=False, return_tensors="np",
                           split_special_tokens=plain_text)
        if result["input_ids"].shape[1] > MAX_TEXT_TOKENS:
            raise RuntimeError("Reference input exceeds the 8192-token context")
        return {key: np.asarray(result[key], dtype=np.int64)
                for key in ("input_ids", "attention_mask")}

    def infer(inputs, image_features=empty, video_features=empty, audio_features=empty):
        if args.preprocess_only:
            return [], 0.0
        feed = dict(inputs, image_features=image_features,
                    video_features=video_features, audio_features=audio_features)
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = model.run(["sentence_embedding"], feed)[0]
            durations.append((time.perf_counter() - start) * 1000)
        if result.ndim != 2 or result.shape[1] != NATIVE_EMBEDDING_DIM or not np.isfinite(result).all():
            raise RuntimeError("Reference returned invalid embeddings")
        norms = np.linalg.norm(result, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
            raise RuntimeError("Reference returned non-unit embeddings")
        reference = result.tolist()
        if not args.native_output:
            reference = [project_indexed_vector(vector) for vector in reference]
        return reference, statistics.median(durations)

    texts = ["a red dress outdoors", "海边的日落", "Mixed CASE café portrait",
             "a photo of " + "a colorful tree beside a lake " * 30,
             "?!", "  multiple   spaces\tand\nlines  ", "ΟΣ İSTANBUL STRAẞE",
             "👩🏽‍💻 café e\u0301 東京 العربية",
             "literal <|image|> <|audio|> <|video|> tokens in documentation"]
    for i, text in enumerate(texts):
        name = f"text-{i}"
        inputs = tokens([QUERY_PREFIX + text], plain_text=True)
        reference, elapsed = infer(inputs)
        fixtures.append(dict(
            name=name, kind="text", text=text,
            input_file=write_tensor(name, "ids", inputs["input_ids"].astype("<i8")),
            shape=list(inputs["input_ids"].shape),
            attention_file=write_tensor(name, "attention", inputs["attention_mask"].astype("<i8")),
            reference=reference, reference_median_ms=elapsed,
        ))
        print(name, "reference_ms=", round(elapsed, 3), flush=True)

    documents = [
        ("none", "The northern lights are caused by charged particles from the sun."),
        ("Mountain photos", "A red bicycle stands beside a mountain lake."),
        ("多言語", "海边的日落。Mixed CASE café portrait and green hills."),
        ("Special <|image|> tokens", "Markdown containing <|audio|> and <|video|> examples."),
    ]
    for i, (title, text) in enumerate(documents):
        name = f"document-{i}"
        inputs = tokens([f"title: {title} | text: {text}"], plain_text=True)
        reference, elapsed = infer(inputs)
        fixtures.append(dict(
            name=name, kind="document", title=title, text=text,
            input_file=write_tensor(name, "ids", inputs["input_ids"].astype("<i8")),
            shape=list(inputs["input_ids"].shape),
            attention_file=write_tensor(name, "attention", inputs["attention_mask"].astype("<i8")),
            reference=reference, reference_median_ms=elapsed,
        ))

    def image(width, height):
        yy, xx = np.indices((height, width))
        pixels = np.stack([(xx * 3 + yy) % 256, (xx + yy * 2) % 256,
                           ((xx // 64 + yy // 64) % 2) * 255], axis=-1).astype(np.uint8)
        return Image.fromarray(pixels)

    sizes = [(1, 1), (7, 1537), (1537, 7), (640, 480), (480, 640),
             (512, 512), (736, 1181), (1600, 900), (48, 48), (49, 48), (48, 49)]
    cases = [(f"image-{w}x{h}", [image(w, h)]) for w, h in sizes]
    cases += [("image-batch2-mixed", [image(640, 480), image(512, 512)]),
              ("image-batch8-mixed", [image(*sizes[3 + i % 3]) for i in range(8)])]
    cases = [(name, images) for name, images in cases if len(images) <= args.max_image_batch]
    sources = {}
    for name, images in cases:
        sources[name] = []
        for index, source in enumerate(images):
            filename = f"{name}-source-{index}.png"
            source.save(output / filename)
            sources[name].append(filename)
    for label, source, extension, save_options in [
        ("jpeg", image(736, 1181), "jpg", {"quality": 91}),
        ("jpeg-gray", image(320, 160).convert("L"), "jpg", {}),
        ("webp", image(736, 1181), "webp", {"quality": 85}),
        ("webp-misnamed", image(736, 1181), "jpg", {"format": "WEBP", "quality": 85}),
        ("png-alpha", image(241, 483).convert("RGBA"), "png", {}),
        ("png-gray", image(99, 193).convert("L"), "png", {}),
    ]:
        name = "decode-" + label
        filename = f"{name}.{extension}"
        source.save(output / filename, **save_options)
        with Image.open(output / filename) as encoded:
            decoded = encoded.convert("RGB")
            decoded.load()
        cases.append((name, [decoded]))
        sources[name] = [filename]

    for name, images in cases:
        with torch.inference_mode():
            prepared = processor(images=images, return_tensors="pt")
        pixels = prepared["pixel_values"].cpu().numpy().astype("<f4")
        positions = prepared["image_position_ids"].cpu().numpy().astype("<i8")
        soft = [int(n) for n in prepared["num_soft_tokens_per_image"]]
        expanded = [tokenizer.boi_token + tokenizer.image_token * n + tokenizer.eoi_token for n in soft]
        inputs = tokens(expanded)
        for ids, n in zip(inputs["input_ids"], soft):
            if np.count_nonzero(ids == 258880) != n:
                raise RuntimeError("Reference tokenizer did not preserve image soft tokens")
        features = empty
        if not args.preprocess_only:
            if vision is None:
                vision = ort.InferenceSession(str(models / "onnx/vision_encoder.onnx"), options,
                                             providers=["CPUExecutionProvider"])
            features = vision.run(["image_features"], {
                "pixel_values": pixels, "pixel_position_ids": positions,
            })[0]
            if features.shape != (sum(soft), 512):
                raise RuntimeError("Reference vision graph returned the wrong number of soft tokens")
        reference, elapsed = infer(inputs, features)
        spatial = []
        for pos in positions:
            active = pos[pos[:, 0] >= 0]
            spatial.append([int(active[:, 1].max() + 1), int(active[:, 0].max() + 1)])
        fixtures.append(dict(
            name=name, kind="image", image_files=sources[name],
            input_file=write_tensor(name, "pixels", pixels), shape=list(pixels.shape),
            position_reference_file=write_tensor(name, "positions", positions),
            spatial_shapes=spatial, num_soft_tokens=soft,
            token_ids_file=write_tensor(name, "ids", inputs["input_ids"].astype("<i8")),
            token_shape=list(inputs["input_ids"].shape),
            attention_file=write_tensor(name, "attention", inputs["attention_mask"].astype("<i8")),
            reference=reference, reference_median_ms=elapsed,
        ))
        print(name, "soft_tokens=", soft, "reference_ms=", round(elapsed, 3), flush=True)

    # Frames are already sampled here. Decoder/FPS policy is validated separately;
    # these fixtures isolate the 140-soft-token budget and video placeholder path.
    for count in [1, 4]:
        name = f"video-frames-{count}"
        images = [image(640, 480) if i % 2 == 0 else image(640, 480).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                  for i in range(count)]
        files = []
        for i, frame in enumerate(images):
            filename = f"{name}-source-{i}.png"
            frame.save(output / filename)
            files.append(filename)
        with torch.inference_mode():
            prepared = video_processor(images=images, return_tensors="pt")
        pixels = prepared["pixel_values"].cpu().numpy().astype("<f4")
        positions = prepared["image_position_ids"].cpu().numpy().astype("<i8")
        soft = [int(n) for n in prepared["num_soft_tokens_per_image"]]
        expanded = "".join(tokenizer.boi_token + "<|video|>" * n + tokenizer.eoi_token for n in soft)
        inputs = tokens([expanded])
        features = empty
        if not args.preprocess_only:
            if vision is None:
                vision = ort.InferenceSession(str(models / "onnx/vision_encoder.onnx"), options,
                                             providers=["CPUExecutionProvider"])
            features = vision.run(["image_features"], {
                "pixel_values": pixels, "pixel_position_ids": positions,
            })[0]
            if features.shape != (sum(soft), 512):
                raise RuntimeError("Video vision-feature count mismatch")
        reference, elapsed = infer(inputs, video_features=features)
        spatial = []
        for pos in positions:
            active = pos[pos[:, 0] >= 0]
            spatial.append([int(active[:, 1].max() + 1), int(active[:, 0].max() + 1)])
        fixtures.append(dict(
            name=name, kind="video", image_files=files,
            input_file=write_tensor(name, "pixels", pixels), shape=list(pixels.shape),
            position_reference_file=write_tensor(name, "positions", positions),
            spatial_shapes=spatial, num_soft_tokens=soft,
            token_ids_file=write_tensor(name, "ids", inputs["input_ids"].astype("<i8")),
            token_shape=list(inputs["input_ids"].shape),
            attention_file=write_tensor(name, "attention", inputs["attention_mask"].astype("<i8")),
            reference=reference, reference_median_ms=elapsed,
        ))
        print(name, "soft_tokens=", soft, flush=True)

    # Free the image/video encoder before loading the independent audio encoder.
    if vision is not None:
        del vision
        import gc
        gc.collect()
    if not args.preprocess_only:
        audio_encoder = ort.InferenceSession(str(models / "onnx/audio_encoder.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    audio_cases = {
        "silence-1s": np.zeros(16000, dtype=np.float32),
        "ramp-161": np.linspace(-0.5, 0.5, 161, dtype=np.float32),
        "sine-1s": (0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32),
        "sine-16001": (0.25 * np.sin(2 * np.pi * 440 * np.arange(16001) / 16000)).astype(np.float32),
        "impulse-2s": np.zeros(32000, dtype=np.float32),
    }
    audio_cases["impulse-2s"][16000] = 0.5
    for label, samples in audio_cases.items():
        name = "audio-" + label
        prepared = audio_processor([samples], sampling_rate=16000, truncation=False,
                                   padding="longest", pad_to_multiple_of=128, return_tensors="np")
        features = np.asarray(prepared["input_features"], dtype=np.float32)
        mask = np.asarray(prepared["input_features_mask"], dtype=np.bool_)
        soft = int(mask[0, ::4].sum())
        if soft == 0:
            raise RuntimeError("Audio fixture has no valid soft tokens")
        inputs = tokens([tokenizer.boa_token + tokenizer.audio_token * soft + tokenizer.eoa_token])
        encoded = empty
        if not args.preprocess_only:
            encoded = audio_encoder.run(["audio_features"], {
                "input_features": features, "input_features_mask": mask,
            })[0]
            if encoded.shape != (soft, 512):
                raise RuntimeError("Audio encoder soft-token count mismatch")
        reference, elapsed = infer(inputs, audio_features=encoded)
        fixtures.append(dict(
            name=name, kind="audio", audio_file=write_tensor(name, "pcm", samples.astype("<f4")),
            audio_samples=len(samples),
            input_file=write_tensor(name, "features", features.astype("<f4")), shape=list(features.shape),
            audio_mask_file=write_tensor(name, "frame-mask", mask.astype(np.uint8)),
            num_soft_tokens=[soft],
            token_ids_file=write_tensor(name, "ids", inputs["input_ids"].astype("<i8")),
            token_shape=list(inputs["input_ids"].shape),
            attention_file=write_tensor(name, "attention", inputs["attention_mask"].astype("<i8")),
            reference=reference, reference_median_ms=elapsed,
        ))
        print(name, "frames=", features.shape[1], "soft_tokens=", soft, flush=True)

    (output / "fixtures.json").write_text(json.dumps(fixtures, indent=2, allow_nan=False) + "\n")
    (output / "environment.json").write_text(json.dumps(dict(
        model_repository=manifest["repository"], model_revision=REVISION,
        output_mode="native" if args.native_output else "indexed",
        embedding_dimension=NATIVE_EMBEDDING_DIM if args.native_output else EMBEDDING_DIM,
        native_embedding_dimension=NATIVE_EMBEDDING_DIM,
        model_profile=NATIVE_MODEL_PROFILE if args.native_output else MODEL_PROFILE,
        torch=torch.__version__, torchvision=torchvision.__version__,
        transformers=transformers.__version__, pillow=PIL.__version__,
        numpy=np.__version__, onnxruntime=ort.__version__ if ort else None,
        provider="CPUExecutionProvider" if ort else None, inference_validated=not args.preprocess_only,
        intra_threads=6, interop_threads=1, image_processor=type(processor).__name__,
        max_image_batch=args.max_image_batch,
        actual_backbone_providers=model.get_providers() if ort else [],
        timing_scope="reference_median_ms measures shared-backbone inference only; modality encoders, preprocessing and loading excluded",
        note="Synthetic image, document, audio and sampled-video fixtures; HF preprocessing and pinned ONNX reference. Not decoder, original-checkpoint or semantic-quality validation.",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
