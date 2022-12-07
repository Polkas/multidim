import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Union, Sequence, Any, Optional
from scipy.cluster.hierarchy import dendrogram

SequenceLike = Union[Sequence, np.ndarray, pd.Series]


def f_test(x: SequenceLike, y: SequenceLike) -> Dict[str, float]:
    """
    Calculates the F-test.
    :param x: SequenceLike The first group of data
    :param y: SequenceLike The second group of data
    :return: a dict with the F statistic value and the p-value.

    >>> from multidim.funs import f_test
    >>> f_test([1, 2, 3], [2, 4, 6])
    {'statistic': 0.25, 'pval-greater': 0.8, 'pval-less': 0.2, 'pval-not equal': 1.6}
    >>>
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
    """Correlation matrix between two different data matrices.
    n-th variable in the first matrix with the n-th variable in the second matrix.
    Arguments:
        X1 -- matrix_like first matrix
        X2 -- matrix_like second matrix
    Returns:
        Correlation matrix, a pandas.DataFrame.
    >>> from multidim.funs import corr_mat
    >>> import numpy as np
    >>> corr_mat(np.array([[1, 2], [2, 3], [4, 2]]), np.array([[2, 5], [3, 9], [5, 7]]))
              0         1
    0  1.000000  0.327327
    1 -0.188982  0.866025
    """
    assert (
        X1.shape[0] == X2.shape[0]
    ), "X1 and X2 should have the same number of observations."

    col_nams: List[str] = list()
    if isinstance(X2, pd.DataFrame):
        col_nams = X2.columns
    else:
        col_nams = [str(e) for e in range(X2.shape[1])]

    index_nams: List[str] = list()
    if isinstance(X1, pd.DataFrame):
        index_nams = X1.columns
    else:
        index_nams = [str(e) for e in range(X1.shape[1])]

    X1 = np.array(X1)
    X2 = np.array(X2)

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


def plot_dendrogram(model: Any, **kwargs: Optional[Any]) -> None:
    """Dendogram Plot
    Wrapper around scipy.cluster.hierarchy.dendrogram
    Arguments:
        model -- result of fit method for sklearn.cluster.AgglomerativeClustering
        **kwargs -- additional arguments for scipy.cluster.hierarchy.dendrogram
    Returns:
        None and draw a plot
    Note:
        Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
