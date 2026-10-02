import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from stringart.geometry import circle_pins  # noqa: E402

SIZE = 120


@pytest.fixture
def pins():
    return circle_pins(64, SIZE)


@pytest.fixture
def target():
    """Light disc with a dark horizontal band, white outside the circle."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    c = (SIZE - 1) / 2
    t = np.full((SIZE, SIZE), 0.9)
    t[50:70, :] = 0.2
    t[(xx - c) ** 2 + (yy - c) ** 2 > c**2] = 1.0
    return t
