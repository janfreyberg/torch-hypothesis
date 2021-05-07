import numpy as np
from hypothesis import given
from torch_hypothesis import is_numeric, classification_logits_and_labels, class_logits


def test_is_numeric():
    # numeric stuff
    for v in [0, 0.2, np.float32(0.3), "1e-4"]:
        assert is_numeric(v), v

    # not numeric stuff
    for v in ["hello"]:
        assert not is_numeric(v), v


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_classif_logits(y_ypred):
    predicted, true = y_ypred
    assert 1 <= predicted.shape[0] <= 32 and 1 <= true.shape[0] <= 32
    assert 2 <= predicted.shape[1] <= 50 and 0 <= true.max() <= 48


@given(logits=class_logits(batch_size=(1, 64), n_classes=(2, 50)))
def test_class_logits(logits):
    assert 1 <= logits.shape[0] <= 64
    assert 2 <= logits.shape[1] <= 50
