from multidim.datasets import load_zadowolenie

from multidim.funs import corr_mat
from multidim.redunt import Redunt
import numpy as np
import pandas as pd

from statsmodels.multivariate.cancorr import CanCorr
from sklearn.cross_decomposition import CCA


def test_Redunt():

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
    redun = Redunt(
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
