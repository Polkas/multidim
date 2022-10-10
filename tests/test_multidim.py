from multidim.datasets import (
    load_iris,
    load_uscities,
    load_tibetan,
    load_auto,
    load_seul1988,
)
from multidim.utils import resolve_stata
from pandas import DataFrame
from multidim.funs import f_test
from multidim.funs import corr_mat
import numpy as np
import pandas as pd
import pytest


def test_load_iris():
    assert isinstance(load_iris(), DataFrame)
    assert isinstance(load_uscities(), DataFrame)
    assert isinstance(load_tibetan(), DataFrame)
    assert isinstance(load_auto(), DataFrame)
    assert isinstance(load_seul1988(), DataFrame)


def test_resolve_stata():
    current_stata = resolve_stata()
    assert isinstance(current_stata, tuple)
    assert len(current_stata) == 2


def test_funs_f_test():
    f_t = f_test([1, 2, 3], [2, 4, 6])
    assert f_t == {
        "statistic": 0.25,
        "pval-greater": 0.8,
        "pval-less": 0.2,
        "pval-not equal": 1.6,
    }


def test_corr_mat():
    c_m = corr_mat(
        np.array([[1, 2], [2, 3], [4, 2]]), np.array([[2, 5], [3, 9], [5, 7]])
    )
    assert np.allclose(
        c_m, pd.DataFrame(np.array([[1.000000, 0.327327], [-0.188982, 0.866025]]))
    )
