"""Evaluation-only image content and low-level pattern analysis.

This module measures correlations between image properties and model errors.
It does not claim that a correlated property is an AIGC fingerprint, and its
outputs must never be used to mine official-val or EvalGEN images for training.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.eval.metrics import binary_metrics


FEATURE_VERSION = 1
LOW_LEVEL_SLICES = (
    "aspect",
    "resolution",
    "format",
    "brightness",
    "saturation",
    "sharpness",
    "entropy",
    "black_border",
)

SEMANTIC_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "style": {
        "photoreal": ("a realistic camera photograph", "a photo of a real-world scene"),
        "illustration": ("a digital illustration", "a graphic illustration"),
        "anime": ("an anime or cartoon image",),
        "painting": ("a painting or artistic artwork",),
    },
    "scene": {
        "indoor": ("an indoor scene",),
        "outdoor": ("an outdoor scene",),
        "studio_product": ("a studio product photograph",),
        "screenshot_ui": ("a screenshot of a website or user interface",),
    },
    "subject": {
        "people": ("an image whose main subject is a person or group of people",),
        "animal": ("an image whose main subject is an animal",),
        "food": ("an image whose main subject is food",),
        "architecture": ("an image whose main subject is a building or architecture",),
        "landscape": ("an image whose main subject is a natural landscape",),
        "object_other": ("an image whose main subject is an everyday object",),
    },
    "composition": {
        "close_up": ("a close-up composition",),
        "wide_scene": ("a wide scene composition",),
        "text_heavy": ("an image containing prominent text, signs, or typography",),
        "abstract": ("an abstract or surreal composition",),
    },
}


def _bucket(value: float, low: float, high: float, names: tuple[str, str, str]) -> str:
    if value < low:
        return names[0]
    if value > high:
        return names[2]
    return names[1]


def _entropy(gray: np.ndarray) -> float:
    counts = np.bincount(np.clip(gray * 255, 0, 255).astype(np.uint8).ravel(), minlength=256)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _black_border_ratio(rgb: np.ndarray) -> float:
    height, width = rgb.shape[:2]
    border_y = max(1, round(height * 0.04))
    border_x = max(1, round(width * 0.04))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True
    border = rgb[mask]
    return float((border.max(axis=1) < 0.06).mean())


def extract_low_level(path: Path, max_side: int = 512) -> dict[str, Any]:
    """Extract deterministic low-level features and categorical slices."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    with Image.open(path) as source:
        width, height = source.size
        source_format = (source.format or path.suffix.lstrip(".") or "unknown").lower()
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side))

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    luma = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )
    chroma_max = rgb.max(axis=2)
    chroma_min = rgb.min(axis=2)
    saturation = np.divide(
        chroma_max - chroma_min,
        chroma_max,
        out=np.zeros_like(chroma_max),
        where=chroma_max > 1e-6,
    )
    dx = np.abs(np.diff(luma, axis=1)).ravel()
    dy = np.abs(np.diff(luma, axis=0)).ravel()
    gradients = np.concatenate((dx, dy))
    sharpness = float(gradients.mean()) if gradients.size else 0.0
    edge_density = float((gradients > 0.12).mean()) if gradients.size else 0.0
    entropy = _entropy(luma)
    black_border_ratio = _black_border_ratio(rgb)
    aspect_ratio = width / height
    longest_side = max(width, height)
    brightness = float(luma.mean())
    contrast = float(luma.std())
    mean_saturation = float(saturation.mean())

    if 0.95 <= aspect_ratio <= 1.05:
        aspect = "square"
    elif aspect_ratio < 0.95:
        aspect = "portrait"
    else:
        aspect = "landscape"
    if longest_side < 256:
        resolution = "tiny"
    elif longest_side < 512:
        resolution = "small"
    elif longest_side < 1024:
        resolution = "medium"
    else:
        resolution = "large"

    return {
        "feature_version": FEATURE_VERSION,
        "analysis_max_side": max_side,
        "path": path.resolve().as_posix(),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "longest_side": longest_side,
        "source_format": "jpg" if source_format == "jpeg" else source_format,
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "saturation_mean": mean_saturation,
        "sharpness_mean_gradient": sharpness,
        "edge_density": edge_density,
        "entropy_bits": entropy,
        "black_border_ratio": black_border_ratio,
        "aspect": aspect,
        "resolution": resolution,
        "format": "jpg" if source_format == "jpeg" else source_format,
        "brightness": _bucket(
            brightness, 0.30, 0.70, ("dark", "mid_brightness", "bright")
        ),
        "saturation": _bucket(
            mean_saturation, 0.15, 0.50, ("low_saturation", "mid_saturation", "high_saturation")
        ),
        "sharpness": _bucket(
            sharpness, 0.025, 0.080, ("soft", "normal_sharpness", "sharp")
        ),
        "entropy": _bucket(entropy, 5.0, 7.0, ("low_entropy", "mid_entropy", "high_entropy")),
        "black_border": "black_border" if black_border_ratio >= 0.50 else "no_black_border",
    }


def load_feature_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a JSONL cache keyed by canonical absolute path."""
    if path is None or not path.is_file():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if not isinstance(row, dict) or "path" not in row:
                raise ValueError(f"{path}:{lineno}: expected an object with path")
            if row.get("feature_version") == FEATURE_VERSION:
                cache[str(row["path"])] = row
    return cache


def write_feature_cache(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row["path"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ClipSemanticExtractor:
    """Optional local-only CLIP zero-shot semantic feature extractor."""

    def __init__(self, model_path: Path, device: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "semantic features require torch and transformers; "
                "run without --semantic if unavailable"
            ) from exc

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.eval()
        self.text_features = self._build_text_features()

    def _encode_prompts(self, prompts: list[str]):
        batch = self.processor(
            text=prompts, padding=True, truncation=True, return_tensors="pt"
        )
        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
            if self.torch.is_tensor(value)
        }
        with self.torch.no_grad():
            features = self.model.get_text_features(**batch)
        return features / features.norm(dim=-1, keepdim=True)

    def _build_text_features(self) -> dict[str, tuple[list[str], Any]]:
        axes: dict[str, tuple[list[str], Any]] = {}
        for axis, labels_to_prompts in SEMANTIC_PROMPTS.items():
            labels = list(labels_to_prompts)
            label_features = []
            for label in labels:
                encoded = self._encode_prompts(list(labels_to_prompts[label]))
                averaged = encoded.mean(dim=0)
                label_features.append(averaged / averaged.norm())
            axes[axis] = (labels, self.torch.stack(label_features))
        return axes

    def extract(self, paths: list[Path], batch_size: int = 32) -> dict[str, dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output: dict[str, dict[str, Any]] = {}
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                    valid_paths.append(path)
                except OSError:
                    continue
            if not images:
                continue
            pixels = self.processor(images=images, return_tensors="pt").pixel_values
            with self.torch.no_grad():
                image_features = self.model.get_image_features(
                    pixel_values=pixels.to(self.device)
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            per_image = [dict() for _ in valid_paths]
            for axis, (labels, text_features) in self.text_features.items():
                similarities = image_features @ text_features.T
                values, indices = similarities.topk(k=min(2, len(labels)), dim=1)
                for index, target in enumerate(per_image):
                    best = int(indices[index, 0])
                    margin = (
                        float(values[index, 0] - values[index, 1])
                        if values.shape[1] > 1
                        else 0.0
                    )
                    target[f"semantic_{axis}"] = labels[best]
                    target[f"semantic_{axis}_score"] = float(values[index, 0])
                    target[f"semantic_{axis}_margin"] = margin
            for path, features in zip(valid_paths, per_image):
                output[path.resolve().as_posix()] = features
        return output


def slice_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = list(LOW_LEVEL_SLICES)
    semantic = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("semantic_")
            and not key.endswith("_score")
            and not key.endswith("_margin")
        }
    )
    return keys + semantic


def _metric_row(
    subset: list[dict[str, Any]],
    threshold: float,
    overall: dict[str, Any],
    total_real: int,
    total_fake: int,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in subset]
    scores = [float(row["pred"]) for row in subset]
    metrics = binary_metrics(labels, scores, threshold)
    n_real = int(metrics["n_real"])
    n_fake = int(metrics["n_fake"])
    fnr = metrics["fn"] / n_fake if n_fake else None
    return {
        **metrics,
        "fnr": fnr,
        "real_prevalence": n_real / total_real if total_real else None,
        "fake_prevalence": n_fake / total_fake if total_fake else None,
        "prevalence_gap": (
            n_fake / total_fake - n_real / total_real
            if total_real and total_fake
            else None
        ),
        "delta_fpr": metrics["fpr"] - overall["fpr"] if n_real else None,
        "delta_fnr": fnr - overall["fnr"] if n_fake else None,
    }


def aggregate_slices(
    rows: list[dict[str, Any]],
    threshold: float = 0.5,
    min_support: int = 20,
    min_generator_support: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate image patterns against predictions and rank candidates."""
    if not rows:
        raise ValueError("cannot aggregate an empty row set")
    if min_support <= 0 or min_generator_support <= 0:
        raise ValueError("support thresholds must be positive")
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["pred"]) for row in rows]
    overall = binary_metrics(labels, scores, threshold)
    overall["fnr"] = overall["fn"] / overall["n_fake"] if overall["n_fake"] else None
    total_real, total_fake = overall["n_real"], overall["n_fake"]

    generator_overall: dict[str, float] = {}
    fake_by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        generator = str(row.get("generator") or "unknown").lower()
        if int(row["label"]) == 1 and generator not in {"real", "unknown"}:
            fake_by_generator[generator].append(row)
    for generator, subset in fake_by_generator.items():
        if len(subset) >= min_generator_support:
            generator_overall[generator] = sum(
                float(row["pred"]) < threshold for row in subset
            ) / len(subset)

    output = []
    for feature in slice_keys(rows):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(feature)
            if value is not None:
                grouped[str(value)].append(row)
        for value, subset in sorted(grouped.items()):
            if len(subset) < min_support:
                continue
            result = _metric_row(
                subset, threshold, overall, total_real=total_real, total_fake=total_fake
            )
            stable_generators = 0
            evaluated_generators = 0
            for generator, base_fnr in generator_overall.items():
                generator_slice = [
                    row
                    for row in subset
                    if int(row["label"]) == 1
                    and str(row.get("generator") or "").lower() == generator
                ]
                if len(generator_slice) < min_generator_support:
                    continue
                evaluated_generators += 1
                slice_fnr = sum(
                    float(row["pred"]) < threshold for row in generator_slice
                ) / len(generator_slice)
                if slice_fnr >= base_fnr + 0.05:
                    stable_generators += 1

            prevalence_gap = result["prevalence_gap"]
            result.update(
                {
                    "feature": feature,
                    "value": value,
                    "generator_count": evaluated_generators,
                    "high_fnr_generators": stable_generators,
                    "shortcut_candidate": bool(
                        prevalence_gap is not None and abs(prevalence_gap) >= 0.15
                    ),
                    "blind_spot_candidate": bool(
                        (result["delta_fpr"] is not None and result["delta_fpr"] >= 0.10)
                        or (result["delta_fnr"] is not None and result["delta_fnr"] >= 0.10)
                        or stable_generators >= 2
                    ),
                }
            )
            output.append(result)

    output.sort(
        key=lambda row: (
            not row["blind_spot_candidate"],
            not row["shortcut_candidate"],
            -max(
                abs(row["delta_fpr"] or 0),
                abs(row["delta_fnr"] or 0),
                abs(row["prevalence_gap"] or 0),
            ),
            -row["n"],
        )
    )
    return output, overall


def aggregate_groups(
    rows: list[dict[str, Any]], field: str, threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Aggregate by condition or generator using shared reals for generators."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if field == "generator":
        reals = [row for row in rows if int(row["label"]) == 0]
        for row in rows:
            if int(row["label"]) == 1:
                grouped[str(row.get("generator") or "unknown")].append(row)
        grouped = {key: reals + values for key, values in grouped.items()}
    else:
        for row in rows:
            grouped[str(row.get(field) or "unknown")].append(row)
    output = []
    for value, subset in sorted(grouped.items()):
        labels = [int(row["label"]) for row in subset]
        scores = [float(row["pred"]) for row in subset]
        metrics = binary_metrics(labels, scores, threshold)
        metrics["fnr"] = (
            metrics["fn"] / metrics["n_fake"] if metrics["n_fake"] else None
        )
        metrics[field] = value
        output.append(metrics)
    return output


def write_slice_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature", "value", "n", "n_real", "n_fake", "acc", "auroc",
        "fpr", "fnr", "mean_pred", "real_prevalence", "fake_prevalence",
        "prevalence_gap", "delta_fpr", "delta_fnr", "generator_count",
        "high_fnr_generators", "shortcut_candidate", "blind_spot_candidate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_slice_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "feature", "value", "n", "n_real", "n_fake", "auroc", "fpr", "fnr",
        "prevalence_gap", "shortcut_candidate", "blind_spot_candidate",
    )

    def format_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(format_value(row.get(column)) for column in columns) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    rows: list[dict[str, Any]],
    overall: dict[str, Any],
    split: str,
    threshold: float,
    top_k: int = 15,
) -> str:
    candidates = [
        row
        for row in rows
        if row["shortcut_candidate"] or row["blind_spot_candidate"]
    ][:top_k]
    lines = [
        "# Content Pattern Analysis",
        "",
        f"> Split: `{split}` · threshold={threshold:.2f} · n={overall['n']}.",
        "> Candidate patterns are correlations for human review, not causal AIGC fingerprints.",
        "> Official val and EvalGEN are evaluation-only and must never feed training or hard-negative mining.",
        "",
        "## Overall",
        "",
        f"- AUROC: {overall['auroc'] if overall['auroc'] is not None else 'N/A'}",
        f"- FPR: {overall['fpr']:.4f}",
        f"- FNR: {overall['fnr']:.4f}" if overall["fnr"] is not None else "- FNR: N/A",
        "",
        "## Priority candidates",
        "",
        "| Feature | Value | N | FPR | FNR | ΔFPR | ΔFNR | Prevalence gap | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        flags = []
        if row["shortcut_candidate"]:
            flags.append("shortcut")
        if row["blind_spot_candidate"]:
            flags.append("blind-spot")
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['feature']} | {row['value']} | {row['n']} | "
            f"{fmt(row['fpr'])} | {fmt(row['fnr'])} | "
            f"{fmt(row['delta_fpr'])} | {fmt(row['delta_fnr'])} | "
            f"{fmt(row['prevalence_gap'])} | {', '.join(flags)} |"
        )
    if not candidates:
        lines.append("| — | No candidate crossed the configured thresholds | — | — | — | — | — | — | — |")
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Large real/fake prevalence gaps indicate possible dataset shortcuts.",
        "- Elevated FPR/FNR identifies a model failure slice, not necessarily its cause.",
        "- A semantic pattern is stronger evidence only when it repeats across generators and transforms.",
        "- Confirm every highlighted pattern in the representative gallery before using it in the Error Analysis Note.",
        "",
    ]
    return "\n".join(lines)


def _thumbnail(path: Path, max_side: int = 240) -> str | None:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=72)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except OSError:
        return None


def render_candidate_gallery(
    prediction_rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    max_patterns: int = 8,
    per_error_type: int = 4,
    threshold: float = 0.5,
) -> str:
    """Render representative FP/FN thumbnails for priority pattern candidates."""
    candidates = [
        row
        for row in slice_rows
        if row["shortcut_candidate"] or row["blind_spot_candidate"]
    ][:max_patterns]
    sections = []
    for candidate in candidates:
        feature, value = candidate["feature"], candidate["value"]
        matching = [
            row for row in prediction_rows if str(row.get(feature)) == str(value)
        ]
        fps = sorted(
            (
                row for row in matching
                if int(row["label"]) == 0 and float(row["pred"]) >= threshold
            ),
            key=lambda row: -float(row["pred"]),
        )[:per_error_type]
        fns = sorted(
            (
                row for row in matching
                if int(row["label"]) == 1 and float(row["pred"]) < threshold
            ),
            key=lambda row: float(row["pred"]),
        )[:per_error_type]
        cards = []
        for error_type, subset in (("FP", fps), ("FN", fns)):
            for row in subset:
                thumb = _thumbnail(Path(str(row["_resolved_path"])))
                image = (
                    f'<img src="data:image/jpeg;base64,{thumb}" alt="{error_type}">'
                    if thumb
                    else "<div class='missing'>image unavailable</div>"
                )
                cards.append(
                    "<figure>"
                    f"{image}<figcaption><b>{error_type}</b> pred={float(row['pred']):.3f}"
                    f"<br><code>{html.escape(str(row['image_path']))}</code></figcaption>"
                    "</figure>"
                )
        sections.append(
            f"<section><h2>{html.escape(feature)} = {html.escape(value)}</h2>"
            f"<p>N={candidate['n']} · shortcut={candidate['shortcut_candidate']} · "
            f"blind-spot={candidate['blind_spot_candidate']}</p>"
            f"<div class='grid'>{''.join(cards) or '<p>No FP/FN at threshold.</p>'}</div></section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Content pattern candidates</title>"
        "<style>body{font:14px sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}"
        "figure{margin:0;border:1px solid #ccc;padding:8px}img{width:100%;height:180px;object-fit:contain}"
        "code{font-size:10px;word-break:break-all}.missing{height:180px;background:#ddd}</style>"
        "</head><body><h1>Content pattern candidates</h1>"
        "<p>Evaluation-only correlations. Human confirmation required.</p>"
        + "".join(sections)
        + "</body></html>"
    )
