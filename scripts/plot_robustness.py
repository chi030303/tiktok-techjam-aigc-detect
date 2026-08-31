#!/usr/bin/env python3
# 2026-09-01, tianqi, repo figures for deliverable 4: clean vs 14 official transforms
"""Write clean-vs-transform figures into docs/robustness/.

  python scripts/plot_robustness.py

Needs matplotlib for PNG. SVG is always written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "robustness" / "official_val400.json"
OUT = REPO / "docs" / "robustness"

INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
CARD = "#F8FAFC"


def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def _svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def write_svg_clean_vs_robust(payload: dict, dest: Path) -> None:
    models = list(payload["models"].items())
    w, h = 920, 340
    left, top, plot_w, plot_h = 70, 48, 720, 230
    y0, y1 = 0.92, 1.0
    n_g = len(models)
    group_w = plot_w / n_g
    bar_w = group_w / 4.2
    keys = (("auc_clean", "#1D4ED8", "AUC clean"), ("auc_robust", "#0F766E", "AUC robust (14)"), ("formula", "#E11D48", "formula"))

    def ypix(v: float) -> float:
        return top + plot_h * (1.0 - (v - y0) / (y1 - y0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="bold" fill="{INK}">',
        "Official val 400  ·  clean vs 14 transforms  ·  score = 0.50 clean + 0.50 robust</text>",
    ]
    for gv in (0.94, 0.96, 0.98, 1.00):
        y = ypix(gv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="11" fill="{MUTED}">{gv:.2f}</text>'
        )
    for i, (name, row) in enumerate(models):
        gx = left + i * group_w + group_w * 0.18
        for j, (key, color, _) in enumerate(keys):
            v = float(row[key])
            x = gx + j * (bar_w + 3)
            y = ypix(v)
            bh = ypix(y0) - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(bh, 1):.1f}" fill="{color}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="10" fill="{INK}">{v:.3f}</text>'
            )
        parts.append(
            f'<text x="{left + (i + 0.5) * group_w:.1f}" y="{top + plot_h + 22:.1f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{INK}">{_svg_escape(name)}</text>'
        )
    lx = left
    for _, color, lab in keys:
        parts.append(f'<rect x="{lx}" y="{h - 28}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{lx + 16}" y="{h - 18}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{INK}">{lab}</text>'
        )
        lx += 170
    parts.append("</svg>")
    dest.write_text("\n".join(parts), encoding="utf-8")


def write_svg_15cond(payload: dict, dest: Path) -> None:
    labels = payload["condition_labels"]
    models = list(payload["models"].items())
    w, h = 1080, 420
    left, top, plot_w, plot_h = 58, 48, 990, 300
    y0, y1 = 0.93, 1.0
    n = len(labels)

    def xpix(i: int) -> float:
        return left + (i / (n - 1)) * plot_w

    def ypix(v: float) -> float:
        return top + plot_h * (1.0 - (v - y0) / (y1 - y0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="bold" fill="{INK}">',
        "AUROC by official condition  ·  fuse stays ≥ 0.984  ·  weakest JPEG-30 / resize ×0.25</text>",
    ]
    for gv in (0.94, 0.96, 0.98, 1.00):
        y = ypix(gv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="11" fill="{MUTED}">{gv:.2f}</text>'
        )
    for i, lab in enumerate(labels):
        x = xpix(i)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{LINE}" stroke-width="0.6"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{MUTED}">{lab}</text>'
        )
    for name, row in models:
        pts = " ".join(f"{xpix(i):.1f},{ypix(v):.1f}" for i, v in enumerate(row["auroc"]))
        color = row["color"]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{pts}"/>'
        )
        for i, v in enumerate(row["auroc"]):
            parts.append(f'<circle cx="{xpix(i):.1f}" cy="{ypix(v):.1f}" r="3.2" fill="{color}"/>')
    lx = left
    for name, row in models:
        parts.append(f'<rect x="{lx}" y="{h - 28}" width="12" height="12" fill="{row["color"]}" rx="2"/>')
        parts.append(
            f'<text x="{lx + 16}" y="{h - 18}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{INK}">'
            f'{_svg_escape(name)}  {row["formula"]:.3f}</text>'
        )
        lx += 230
    parts.append("</svg>")
    dest.write_text("\n".join(parts), encoding="utf-8")


def write_png(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(payload["models"].items())
    labels = payload["condition_labels"]
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), gridspec_kw={"height_ratios": [1.05, 1.35]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    names = [m[0] for m in models]
    x = np.arange(len(names))
    w = 0.26
    clean = [m[1]["auc_clean"] for m in models]
    robust = [m[1]["auc_robust"] for m in models]
    formula = [m[1]["formula"] for m in models]
    ax.bar(x - w, clean, w, color="#1D4ED8", label="AUC clean", edgecolor="white")
    ax.bar(x, robust, w, color="#0F766E", label="AUC robust (14 keys)", edgecolor="white")
    ax.bar(x + w, formula, w, color="#E11D48", label="formula  0.50+0.50", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.94, 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="lower right", ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Clean vs transformed  ·  official val 400 (200 COCO / 200 DALL·E)", loc="left", color=INK)
    for i, (c, r, f) in enumerate(zip(clean, robust, formula)):
        ax.text(i - w, c + 0.0015, f"{c:.3f}", ha="center", fontsize=8, color=INK)
        ax.text(i, r + 0.0015, f"{r:.3f}", ha="center", fontsize=8, color=INK)
        ax.text(i + w, f + 0.0015, f"{f:.3f}", ha="center", fontsize=8, color=INK)

    ax = axes[1]
    xs = np.arange(len(labels))
    for name, row in models:
        ax.plot(xs, row["auroc"], "o-", color=row["color"], lw=2.0, ms=5, label=f"{name}  {row['formula']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.93, 1.005)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(4, color="#E11D48", ls=":", lw=1, alpha=0.5)
    ax.axvline(9, color="#E11D48", ls=":", lw=1, alpha=0.5)
    ax.set_title("15 official conditions  ·  weakest keys JPEG-30 and resize ×0.25", loc="left", color=INK)

    fig.tight_layout()
    dest = OUT / "clean_vs_transforms.png"
    fig.savefig(dest, dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = _load()
    write_svg_clean_vs_robust(payload, OUT / "clean_vs_robust.svg")
    write_svg_15cond(payload, OUT / "auroc_15cond.svg")
    try:
        write_png(payload)
        png = " + clean_vs_transforms.png"
    except Exception as exc:
        png = f" (PNG skipped: {exc})"
    print(f"wrote {OUT}  svg + json{png}")


if __name__ == "__main__":
    main()
# end
