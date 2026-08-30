# 2026-08-30, yun, pure-logic helpers for the SID online-aug engine (exp4 image_size wiring)
from src.train.sid_online import (
    _aug_expand,
    _head_kind,
    _image_size,
    _input_mode,
    _interpolate_pos,
    _sid_feat_cache_path,
)


def test_image_size_defaults_to_224_and_marks_interpolation_only_when_changed():
    assert _image_size({}) == 224
    assert _interpolate_pos({}) is False
    assert _image_size({"image_size": 336}) == 336
    assert _interpolate_pos({"image_size": 336}) is True


def test_head_kind_and_input_mode_default():
    assert _head_kind({}) == "linear"
    assert _head_kind({"head": "mlp"}) == "mlp"
    assert _input_mode({}) == "rgb"
    assert _input_mode({"input_mode": "highpass"}) == "highpass"


def test_aug_expand_recognizes_six_ops_only():
    assert _aug_expand({}) is False
    assert _aug_expand({"aug": {"expand": "six_ops"}}) is True
    assert _aug_expand({"aug": {"expand": "something_else"}}) is False


def test_sid_feat_cache_path_is_keyed_by_backbone_split_seed_size():
    a = _sid_feat_cache_path("clip-vit-base-patch16", "validation", 20000, 1, 224)
    b = _sid_feat_cache_path("clip-vit-base-patch16", "validation", 20000, 1, 224)
    c = _sid_feat_cache_path("clip-vit-base-patch16", "validation", 20000, 1, 336)
    assert a == b
    assert a != c
    assert a.name == "sid_validation_n20000_seed1_s224.pt"
    assert "_featcache_sid_online" in str(a)
# end
