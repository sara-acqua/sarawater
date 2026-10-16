import sys
import os

import pytest

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.utils import _validate_weights


def test_validate_weights_default_and_valid():
    assert _validate_weights(None, 4) == (0.25,) * 4
    assert _validate_weights([0.5, 0.3, 0.2], 3) == (0.5, 0.3, 0.2)


@pytest.mark.parametrize(
    "bad", [[0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 1.0, 0.5], [1, 0, "a"], [1, 0, None]]
)
def test_validate_weights_rejects_invalid(bad):
    with pytest.raises(ValueError, match="weights"):
        _validate_weights(bad, 3)
