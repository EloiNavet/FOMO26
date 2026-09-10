import pytest
import torch
from asparagus.functional.representations import build_h_global, ordered_feature_sequence


def test_ordered_feature_sequence_preserves_sequence_order():
    first = torch.zeros(1, 1, 2, 2, 2)
    second = torch.ones(1, 1, 2, 2, 2)

    assert tuple(id(tensor) for tensor in ordered_feature_sequence([first, second])) == (id(first), id(second))
    assert tuple(id(tensor) for tensor in ordered_feature_sequence((second, first))) == (id(second), id(first))


def test_ordered_feature_sequence_sorts_dict_shallow_to_deep():
    h1 = torch.full((1, 1, 2, 2, 2), 1.0)
    h2 = torch.full((1, 1, 2, 2, 2), 2.0)
    h3 = torch.full((1, 1, 2, 2, 2), 3.0)

    ordered = ordered_feature_sequence({"encoder_3": h3, "h1": h1, "stage2": h2})

    assert tuple(id(tensor) for tensor in ordered) == (id(h1), id(h2), id(h3))


def test_ordered_feature_sequence_rejects_ambiguous_dict_keys():
    feat = torch.zeros(1, 1, 2, 2, 2)

    with pytest.raises(ValueError, match="ending in an integer"):
        ordered_feature_sequence({"stem": feat})
    with pytest.raises(ValueError, match="duplicate"):
        ordered_feature_sequence({"h1": feat, "stage1": feat})


def test_build_h_global_concatenates_pooled_5d_features_and_preserves_gradients():
    first = torch.arange(2 * 3 * 2 * 2 * 2, dtype=torch.float64).reshape(2, 3, 2, 2, 2).requires_grad_()
    second = torch.arange(2 * 5 * 1 * 2 * 3, dtype=torch.float64).reshape(2, 5, 1, 2, 3).requires_grad_()

    h_global = build_h_global([first, second])
    expected = torch.cat(
        [
            first.mean(dim=(2, 3, 4)),
            second.mean(dim=(2, 3, 4)),
        ],
        dim=1,
    )

    assert h_global.shape == (2, 8)
    assert h_global.dtype == first.dtype
    assert h_global.device == first.device
    assert torch.equal(h_global, expected)

    h_global.sum().backward()
    assert first.grad is not None
    assert second.grad is not None


def test_build_h_global_accepts_single_5d_tensor():
    feat = torch.randn(2, 4, 3, 3, 3)

    h_global = build_h_global(feat)

    assert h_global.shape == (2, 4)


def test_build_h_global_rejects_missing_empty_non5d_and_batch_mismatch():
    with pytest.raises(ValueError, match="None"):
        build_h_global(None)
    with pytest.raises(ValueError, match="empty"):
        build_h_global([])
    with pytest.raises(ValueError, match="5D"):
        build_h_global([torch.randn(2, 4)])
    with pytest.raises(ValueError, match="batch size"):
        build_h_global([torch.randn(2, 4, 1, 1, 1), torch.randn(3, 4, 1, 1, 1)])
