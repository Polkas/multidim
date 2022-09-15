from multidim.datasets import load_iris

from pandas import DataFrame

def test_load_iris():
    assert isinstance(load_iris(), DataFrame)