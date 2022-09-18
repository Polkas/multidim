from multidim.datasets import load_iris
from multidim.utils import resolve_stata

from pandas import DataFrame


def test_load_iris():
    assert isinstance(load_iris(), DataFrame)


def test_resolve_stata():
    current_stata = resolve_stata()
    assert isinstance(current_stata, tuple)
    assert len(current_stata) == 2
