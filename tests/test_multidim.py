from multidim.datasets import (
    load_iris,
    load_uscities,
    load_tibetan,
    load_auto,
    load_seul1988,
    load_zadowolenie,
)
from multidim.utils import resolve_stata, overwrite_stata_magic, load_stata
from pandas import DataFrame
from multidim.funs import f_test, REDUNT, corr_mat
import numpy as np
import pandas as pd
from multidim import copy
import os
from unittest.mock import patch
import tempfile
import shutil
import pytest


def test_load_datasets():
    assert isinstance(load_iris(), DataFrame)
    assert isinstance(load_uscities(), DataFrame)
    assert isinstance(load_tibetan(), DataFrame)
    assert isinstance(load_auto(), DataFrame)
    assert isinstance(load_seul1988(), DataFrame)


def test_resolve_stata():
    current_stata = resolve_stata()
    assert isinstance(current_stata, tuple)
    assert len(current_stata) == 2


def test_overwrite_stata_magic():
    from IPython.testing.globalipapp import get_ipython

    ip = get_ipython()
    assert overwrite_stata_magic() is None
    msg = "Stata was not loaded properly. Please check the STATA path and type."
    overwrite_stata_magic()
    assert ip.run_line_magic("stata", "display Hello") == msg
    assert ip.run_cell_magic("stata", "", "display Hello") == msg


def test_load_stata():
    load_stata(None, None)


def test_copy():
    with patch("sys.argv", ["copy", "WRONGPATH"]):
        with pytest.raises(TypeError):
            copy()
    dirpath = tempfile.mkdtemp()
    with patch("sys.argv", ["copy", dirpath]):
        copy()
        assert os.path.isdir(os.path.join(dirpath, "notebooks"))
        assert os.path.isfile(
            os.path.join(dirpath, "notebooks", "02_stats_intro.ipynb")
        )
    shutil.rmtree(dirpath)


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


def test_REDUNDANT():

    from statsmodels.multivariate.cancorr import CanCorr
    from sklearn.cross_decomposition import CCA

    y_vars = ["rodzina", "przyjaciele", "zdrowie", "sukces"]
    cat_vars = ["plec", "zaleznosc", "stanciv", "zaufanie"]
    x_vars = [
        "dep_wyglad",
        "dep_zapal",
        "dep_zdrowie",
        "dep_sen",
        "dep_meczenie",
        "plec_1",
        "wiek2011",
        "kontakty",
        "aktywnosc",
        "edukacja",
        "zaufanie_1",
        "zaleznosc_1",
        "stanciv_1",
        "stanciv_2",
        "stanciv_3",
        "stanciv_4",
        "dochod",
    ]

    zadowolenie = load_zadowolenie()
    zadowolenie = zadowolenie.apply(
        lambda x: x.cat.codes if x.dtypes == "category" else x
    )
    zadowolenie[cat_vars] = zadowolenie[cat_vars].astype("int")
    zadowolenie_cols = pd.get_dummies(
        zadowolenie,
        columns=cat_vars,
        drop_first=True,
    )
    norm_cols = y_vars + [
        "wiek2011",
        "kontakty",
        "aktywnosc",
        "edukacja",
        "dochod",
        "dep_wyglad",
        "dep_zapal",
        "dep_zdrowie",
        "dep_sen",
        "dep_meczenie",
    ]
    zadowolenie_cols[norm_cols] = (
        zadowolenie_cols[norm_cols] - zadowolenie_cols[norm_cols].mean()
    ) / zadowolenie_cols[norm_cols].std(ddof=1)
    y_cols = y_vars
    x_cols = x_vars
    y_mat = zadowolenie_cols[y_cols]
    x_mat = zadowolenie_cols[x_cols]
    my_cca = CCA(n_components=2, copy=True, max_iter=100, scale=False)
    my_cca.fit(x_mat, y_mat)
    x_scores, y_scores = my_cca.transform(x_mat, y_mat)
    scores_corr_X_yscores = corr_mat(x_mat, y_scores)
    scores_corr_Y_xscores = corr_mat(y_mat, x_scores)
    res = CanCorr(y_mat, x_mat)
    redun = REDUNT(
        x_mat, y_mat, res.cancorr, scores_corr_Y_xscores, scores_corr_X_yscores
    )
    assert np.allclose(
        redun[0],
        [
            [0.38735564940549644, 0.17256807822444525],
            [0.21472674882208714, 0.09566139656019419],
        ],
    )
    assert np.allclose(
        redun[1],
        [
            [0.28365240900750716, 0.05123710058229869],
            [0.0623265715688049, 0.01125826086792084],
        ],
    )
