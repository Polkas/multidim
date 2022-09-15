from importlib_resources import files, as_file
from pandas import read_pickle, DataFrame, read_stata
import multidim.data

__all__ = ["load_iris",]


def _get_file_path(file: str):
    """Get a file path"""
    source = files(multidim.data).joinpath(file)
    return source


def load_iris() -> DataFrame:
    """load iris dataset
    Returns:
        pandas.DataFrame: iris dataset
    """
    sour = _get_file_path("iris.dta")
    with as_file(sour) as fil:
        return read_stata(fil)