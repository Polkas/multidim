import numpy as np
import pandas as pd

from typing import List, Dict, Union, Sequence, Any, Optional

SequenceLike = Union[Sequence, np.ndarray, pd.Series]


def Redunt(
    matX: Union[pd.DataFrame, np.ndarray],
    matY: Union[pd.DataFrame, np.ndarray],
    can_corrs: SequenceLike,
    corr_Y_xscores: Union[pd.DataFrame, np.ndarray],
    corr_X_yscores: Union[pd.DataFrame, np.ndarray],
) -> List[pd.DataFrame]:
    """Redundancy for CCA analysis.
    Arguments:
        matX -- matrix_like egzo variables
        matY -- matrix_like engo variables
        can_corrs -- SequenceLike 1D canonical correlations
        corr_Y_xscores -- matrix_like correlation between endo variables and xscores
        corr_X_yscores -- matrix_like correlation between egzo variables and yscores
    Returns:
        Redundancy (List[pd.DataFrame]) - Average percent of variance in a set of variables explained by their own canonical variate.
    """
    assert (
        matX.shape[0] == matY.shape[0]
    ), "matX and matY should have the same number of observations."
    assert (
        corr_Y_xscores.shape[1] == corr_X_yscores.shape[1]
    ), "corr_Y_xscores and corr_X_yscores should have the same number of columns."
    assert (
        len(can_corrs) >= corr_Y_xscores.shape[1]
    ), "can_corrs should have number of elements at least as number of columns in corr_Y_xscores."
    assert (
        len(can_corrs) >= corr_X_yscores.shape[1]
    ), "can_corrs should have number of elements at least as number of columns in corr_X_xscores."

    matX = np.array(matX)
    matY = np.array(matY)
    can_corrs = np.array(can_corrs)
    corr_Y_xscores = np.array(corr_Y_xscores)
    corr_X_yscores = np.array(corr_X_yscores)

    eigenmatY = can_corrs
    vector1 = np.power(eigenmatY, 2)
    names1 = ["own variance", "opposite variance"]
    names2 = ["y", "x"]
    matim = list()
    for i in range(corr_Y_xscores.shape[1]):
        a = np.sum(np.power(corr_Y_xscores[:, i], 2)) / matY.shape[1]
        b = a / vector1[i]
        c = np.sum(np.power(corr_X_yscores[:, i], 2)) / matX.shape[1]
        d = c / vector1[i]
        mm = pd.DataFrame([[b, a], [d, c]])
        mm.index = names2
        mm.columns = names1
        matim.append(mm)
    return matim
