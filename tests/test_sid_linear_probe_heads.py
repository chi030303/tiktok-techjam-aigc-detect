# 2026-08-30, yun, build_head shapes; no backbone weights needed, but the module
# imports transformers at load time (same as src/models/linear_probe.py), so this
# needs the project's torch+transformers env, not a bare `pip install -r requirements.txt`.
import pytest
import torch

from src.models.sid_linear_probe import build_head


def test_linear_head_is_a_single_logit():
    head = build_head(768, "linear")
    out = head(torch.randn(4, 768))
    assert out.shape == (4, 1)


def test_mlp_head_hidden_and_output_shape():
    head = build_head(1024, "mlp")
    out = head(torch.randn(4, 1024))
    assert out.shape == (4, 1)
    assert isinstance(head[0], torch.nn.Linear)
    assert head[0].out_features == 256


def test_unknown_head_kind_rejected():
    with pytest.raises(SystemExit):
        build_head(768, "transformer")
# end
