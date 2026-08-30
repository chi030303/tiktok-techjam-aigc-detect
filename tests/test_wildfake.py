# 2026-08-30, samily, WildFake rule matching tests
from pathlib import Path

from PIL import Image

from src.data.wildfake import load_generators_config, scan_wildfake


def test_wildfake_scan_ddpm_and_real(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs/wildfake/generators.yaml"
    cfg = load_generators_config(cfg_path)
    ddpm = tmp_path / "cross_arch" / "ddpm"
    real = tmp_path / "real" / "ffhq"
    ddpm.mkdir(parents=True)
    real.mkdir(parents=True)
    Image.new("RGB", (512, 512), (1, 2, 3)).save(ddpm / "fake001.png")
    Image.new("RGB", (600, 400), (4, 5, 6)).save(real / "real001.jpg")

    records, stats = scan_wildfake(
        tmp_path,
        cfg,
        generators={"ddpm"},
        min_side=512,
        max_per_generator=10,
        compute_phash=True,
    )
    assert stats["unmatched"] == 0
    assert len(records) == 1
    assert records[0].generator == "ddpm"
    assert records[0].arch == "pixel"
    assert records[0].family == "diffusion"

    all_recs, _ = scan_wildfake(tmp_path, cfg, min_side=0, compute_phash=False)
    assert len(all_recs) == 2
    reals = [r for r in all_recs if r.label == 0]
    assert len(reals) == 1
    assert reals[0].content_type == "real"


def test_force_generator_matches_actual_extracted_root(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs/wildfake/generators.yaml"
    cfg = load_generators_config(cfg_path)
    root = tmp_path / "cross_arch" / "ddpm"
    nested = root / "images" / "class_0"
    nested.mkdir(parents=True)
    Image.new("RGB", (256, 256)).save(nested / "0001.png")

    records, stats = scan_wildfake(
        root,
        cfg,
        force_generator="ddpm",
        min_side=256,
        compute_phash=False,
    )
    assert stats["unmatched"] == 0
    assert len(records) == 1
    assert records[0].generator == "ddpm"


def test_wildfake_real_coco_is_excluded_at_relative_root(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs/wildfake/generators.yaml"
    cfg = load_generators_config(cfg_path)
    folder = tmp_path / "real" / "coco"
    folder.mkdir(parents=True)
    Image.new("RGB", (256, 256)).save(folder / "0001.jpg")
    records, _ = scan_wildfake(tmp_path, cfg, min_side=0, compute_phash=False)
    assert records == []
