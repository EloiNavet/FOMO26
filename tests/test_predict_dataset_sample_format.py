"""Prediction must be able to read the sample format this repository actually writes.

A preprocessed asparagus sample on disk is ``[image, label]``. A prediction path that assumes the
file *is* the image cannot load its own corpus -- and must never consume the label half.
"""

from __future__ import annotations

import pytest
import torch
from asparagus.modules.datasets.TrainDataset import _image_from_sample


def test_a_stored_image_label_pair_yields_the_image():
    image = torch.ones(4, 8, 8, 4)
    label = torch.zeros(1)
    assert _image_from_sample([image, label], "scan.pt") is image
    assert _image_from_sample((image, label), "scan.pt") is image


def test_a_bare_tensor_is_passed_through_unchanged():
    image = torch.ones(1, 8, 8, 8)
    assert _image_from_sample(image, "scan.npy") is image


def test_a_mapping_sample_uses_its_image_entry():
    image = torch.ones(2, 4, 4, 4)
    assert _image_from_sample({"image": image, "label": torch.zeros(1)}, "s.pt") is image
    assert _image_from_sample({"data": image}, "s.pt") is image


@pytest.mark.parametrize("bad", [[], ()])
def test_an_empty_sample_fails_loudly(bad):
    with pytest.raises(ValueError, match="no image"):
        _image_from_sample(bad, "scan.pt")


def test_a_mapping_without_an_image_fails_loudly():
    with pytest.raises(ValueError, match="no 'image'/'data' key"):
        _image_from_sample({"label": torch.zeros(1)}, "scan.pt")


def test_the_label_half_is_never_returned():
    """Regression guard: inference must not be able to read the target."""
    image, label = torch.ones(1, 2, 2, 2), torch.full((1,), 7.0)
    assert not torch.equal(_image_from_sample([image, label], "scan.pt"), label)
