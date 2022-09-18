import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Union, Iterable, Sequence


def f_test(x: Iterable, y: Iterable, alt: str = "two_sided") -> Dict[str, float]:
    """
    Calculates the F-test.
    :param x: Iterable The first group of data
    :param y: Iterable The second group of data
    :param alt: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a dict with the F statistic value and the p-value.
    """
    x = np.array(x)
    y = np.array(y)
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var(ddof=1) / y.var(ddof=1)
    return {
        "statistic": f,
        "pval-greater": 1.0 - stats.f.cdf(f, df1, df2),
        "pval-less": stats.f.cdf(f, df1, df2),
        "pval-not equal": 2.0 * (1.0 - stats.f.cdf(f, df1, df2)),
    }


def corr_mat(
    X1: Union[pd.DataFrame, np.ndarray], X2: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """Correlation matrix between two different matrices.
    Arguments:
        X1 -- matrix_like first matrix
        X2 -- matrix_like second matrix
    Returns:
        Correlation matrix, a pandas.DataFrame.
    """
    assert (
        X1.shape[0] == X2.shape[0]
    ), "X1 and X2 should have the same number of observations."

    col_nams = X2.columns if hasattr(X2, "columns") else list(range(X2.shape[1]))
    index_nams = X1.columns if hasattr(X1, "columns") else list(range(X1.shape[1]))

    X1 = np.asmatrix(X1)
    X2 = np.asmatrix(X2)
    numerator = np.matmul(X1.T, X2) / X2.shape[0] - np.outer(
        np.mean(X1, axis=0), np.mean(X2, axis=0)
    )
    denominator = np.outer(np.std(X1, axis=0), np.std(X2, axis=0))
    res = pd.DataFrame(numerator / denominator)
    res.columns = col_nams
    res.index = index_nams
    return res


def REDUNT(
    matX: Union[pd.DataFrame, np.ndarray],
    matY: Union[pd.DataFrame, np.ndarray],
    can_corrs: Sequence,
    corr_Y_xscores: Union[pd.DataFrame, np.ndarray],
    corr_X_yscores: Union[pd.DataFrame, np.ndarray],
) -> List[pd.DataFrame]:
    """Redundancy for CCA analysis.
    Arguments:
        matX -- matrix_like egzo variables
        matY -- matrix_like engo variables
        can_corrs -- Sequence 1D canonical correlations
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

    matX = np.asmatrix(matX)
    matY = np.asmatrix(matY)
    can_corrs = np.array(can_corrs)
    corr_Y_xscores = np.asmatrix(corr_Y_xscores)
    corr_X_yscores = np.asmatrix(corr_X_yscores)

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
        mm = pd.DataFrame(np.matrix([[b, a], [d, c]]))
        mm.index = names2
        mm.columns = names1
        matim.append(mm)
    return matim
