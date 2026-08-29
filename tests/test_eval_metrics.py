# 2026-08-29, tianqi, unit tests for AUROC / acc / FPR without GPU
from src.eval.metrics import auroc, binary_metrics


def test_perfect_ranking_auroc():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert auroc(y, s) == 1.0
    m = binary_metrics(y, s, threshold=0.5)
    assert m["acc"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0
    assert m["fpr"] == 0.0
    assert m["recall_fake"] == 1.0


def test_auroc_none_when_one_class():
    assert auroc([1, 1], [0.2, 0.9]) is None
    m = binary_metrics([1, 1], [0.2, 0.9], threshold=0.5)
    assert m["n_real"] == 0
    assert m["auroc"] is None
    assert m["recall_fake"] == 0.5


def test_threshold_and_fpr():
    y = [0, 0, 1, 1]
    s = [0.6, 0.1, 0.9, 0.4]
    m = binary_metrics(y, s, threshold=0.5)
    assert m["fp"] == 1 and m["fn"] == 1
    assert m["fpr"] == 0.5
    assert m["precision_fake"] == 0.5


def test_auroc_with_ties():
    y = [0, 1]
    s = [0.5, 0.5]
    assert auroc(y, s) == 0.5
# end
