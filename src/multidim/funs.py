import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Union, Sequence, Any, Optional
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import multivariate_normal

SequenceLike = Union[Sequence, np.ndarray, pd.Series]


def f_test(x: SequenceLike, y: SequenceLike) -> Dict[str, float]:
    """
    Calculates the F-test.
    Arguments:
        x -- SequenceLike The first group of data
        y -- SequenceLike The second group of data
    Returns:
        dict with the F statistic value and the p-value

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

    X1 = np.array(X1, dtype=np.float64)
    X2 = np.array(X2, dtype=np.float64)

    numerator = np.matmul(X1.T, X2) / X2.shape[0] - np.outer(
        np.mean(X1, axis=0), np.mean(X2, axis=0)
    )
    denominator = np.outer(np.std(X1, axis=0), np.std(X2, axis=0))
    res = pd.DataFrame(numerator / denominator)
    res.columns = col_nams
    res.index = index_nams
    return res


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
