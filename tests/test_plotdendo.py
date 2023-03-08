from multidim.funs import plot_dendrogram
from sklearn.cluster import AgglomerativeClustering
from multidim.datasets import load_tibetan


def test_plot_dendrogram():

    tibetan = load_tibetan()
    model_base = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage="single"
    )
    model = model_base.fit(tibetan[["length", "breadth", "height", "upper", "face"]])
    plot_dendrogram(model)
