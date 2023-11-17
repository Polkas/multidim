from multidim.datasets import (
    load_iris,
    load_indeks_spol,
    load_uscities,
    load_tibetan,
    load_auto,
    load_seul1988,
    load_zadowolenie,
    load_nauczyciele,
    load_euro,
    load_depresja,
    load_boston
)
from pandas import DataFrame
from numpy import ndarray


def test_load_datasets():
    assert isinstance(load_iris(), DataFrame)
    assert isinstance(load_uscities(), DataFrame)
    assert isinstance(load_tibetan(), DataFrame)
    assert isinstance(load_auto(), DataFrame)
    assert isinstance(load_seul1988(), DataFrame)
    assert isinstance(load_euro(), DataFrame)
    assert isinstance(load_zadowolenie(), DataFrame)
    assert isinstance(load_nauczyciele(), DataFrame)
    assert isinstance(load_depresja(), DataFrame)
    assert isinstance(load_indeks_spol(), ndarray)
    assert isinstance(load_boston(), DataFrame)
