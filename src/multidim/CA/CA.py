import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import svd_flip


class CaBase:
    """Base correspondence analysis.
    Notes
    -----
    The implementation follows that presented in 'Correspondence
    Analysis in R, with Two- and Three-dimensional Graphics: The ca
    Package,' Journal of Statistical Software, May 2007, Volume 20,
    Issue 3.
    """

    def __init__(self, data, seed=1234):
        assert isinstance(
            data, pd.DataFrame
        ), "data argument has to be a pandas.DataFrame"
        np.random.seed(seed)

    def apply_svd(self, data):
        # contingency table
        N = np.matrix(data, dtype=float)
        # correspondence matrix from contingency table
        P = N / N.sum()
        # row and column marginal totals of P as vectors
        r = P.sum(axis=1)
        c = P.sum(axis=0).T
        # diagonal matrices of row/column sums
        # ipdb.set_trace()
        D_r_rsq = sparse.diags(1.0 / np.sqrt(r.A1))
        D_c_rsq = sparse.diags(1.0 / np.sqrt(c.A1))
        # the matrix of standarized residuals
        S = (D_r_rsq @ (P - np.outer(r, c))) @ D_c_rsq
        # compute the SVD
        U, D_a, Vt = np.linalg.svd(S, full_matrices=False)
        D_a = np.asmatrix(np.diag(D_a))
        U, Vt = svd_flip(U.A, Vt.A)
        V = Vt.T
        # principal coordinates of rows
        F = (P / P.sum(axis=1)) @ D_c_rsq @ V
        # principal coordinates of columns
        G = (P.T / P.T.sum(axis=1)) @ D_r_rsq @ U
        # standard coordinates of rows
        X = D_r_rsq * U
        # standard coordinates of columns
        Y = D_c_rsq * V
        # the total variance of the data matrix
        inertia = np.einsum("ij,ji->", S, S.T)

        self.F = F.A
        self.G = G.A
        self.X = X
        self.Y = Y
        self.inertia = inertia
        self.eigenvals = np.diag(D_a) ** 2


class CaScreeMixin:
    def scree_diagram(self, perc: bool = True, *args, **kwargs) -> None:
        """Plot the scree diagram."""
        eigenvals = self.eigenvals
        xs = np.arange(1, eigenvals.size + 1, 1)
        ys = 100.0 * eigenvals / eigenvals.sum() if perc else eigenvals
        plt.plot(xs, ys, *args, **kwargs)
        plt.xlabel("Dimension")
        plt.ylabel("Eigenvalue" + (" [%]" if perc else ""))


class CaPlotMixin:
    def plot(self):
        fig = plt.figure(figsize=(20, 12))
        ax = fig.gca()
        ax.set(
            xlim=(
                np.min(self.data_out.iloc[:, 0]) * 1.1,
                np.max(self.data_out.iloc[:, 0]) * 1.1,
            ),
            ylim=(
                np.min(self.data_out.iloc[:, 1]) * 1.1,
                np.max(self.data_out.iloc[:, 1]) * 1.1,
            ),
        )
        plt.grid()
        for i, l in enumerate(self.data_out.index):
            ax.text(
                self.data_out.iloc[i, 0],
                self.data_out.iloc[i, 1],
                l,
                color=self.colors[i],
                ha="center",
                va="center",
            )


class CA(CaBase, CaScreeMixin, CaPlotMixin):
    """Correspondence analysis.
    Inputs
    ------
    data : pandas.DataFrame
      Two-way contingency table like returned by pandas.crosstab.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

        self.rows = data.index.values.tolist()
        self.cols = data.columns.values.tolist()
        self.colors = ["b"] * len(self.rows) + ["r"] * len(self.cols)

        self.apply_svd(data)

        """Plot the first and second dimensions."""
        data_r = pd.DataFrame(self.F[:, 0:2])
        data_c = pd.DataFrame(self.G[:, 0:2])
        data_out = pd.concat([data_r, data_c], axis=0)
        data_out.columns = ["Dim 1", "Dim 2"]
        data_out.index = self.rows + self.cols
        self.data_out = data_out


class MCA(CaBase, CaScreeMixin, CaPlotMixin):
    """Multiply Correspondence Analysis.
    Inputs
    ------
    data : pandas.DataFrame
      Original dataset with only binary variables in the long format is expected.
      pandas.get_dummies could be used to get binary variables from categorical ones.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

        data = pd.get_dummies(data)
        self.cols = data.columns.values.tolist()
        cols_prefixes = data.columns.str.split("_").map(lambda x: x[0]).tolist()
        cc_set = set(cols_prefixes)
        cc_set_len = len(cc_set)
        colors_map = dict(
            zip(
                cc_set,
                (list("bgrcmyk") + list("bgrcmyk") * (cc_set_len // 7))[:cc_set_len],
            )
        )
        self.colors = [colors_map.get(i, "b") for i in cols_prefixes]
        self.apply_svd(data)
        data_out = pd.DataFrame(self.G[:, 0:2])
        data_out.columns = ["Dim 1", "Dim 2"]
        data_out.index = self.cols
        self.data_out = data_out
