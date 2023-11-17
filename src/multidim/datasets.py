from importlib_resources import files, as_file
from importlib_resources.abc import Traversable
from pandas import DataFrame, read_stata, read_csv
import multidim.data

__all__ = [
    "load_iris",
    "load_indeks_spol",
    "load_auto",
    "load_uscities",
    "load_tibetan",
    "load_seul1988",
    "load_euro",
    "load_nauczyciele",
    "load_depresja",
    "load_cerealnut",
    "load_boston",
]


def _get_file_path(file: str) -> Traversable:
    """Get a file path"""
    source = files(multidim.data).joinpath(file)
    return source


def load_iris() -> DataFrame:
    """load iris dataset
    Returns:
        pandas.DataFrame: iris dataset
    """
    sour = _get_file_path("iris.csv")
    with as_file(sour) as fil:
        return read_csv(fil)


def load_indeks_spol() -> DataFrame:
    """load iris dataset
    Returns:
        pandas.DataFrame: iris dataset
    """
    sour = _get_file_path("indeks_spol.csv")
    with as_file(sour) as fil:
        return read_csv(fil, header=None).values


def load_uscities() -> DataFrame:
    """load uscities dataset
    Sophia Rabe-Hesketh i Brian Everitt, A Handbook of Statistical Analyses using Stata.
    Returns:
        pandas.DataFrame: uscities dataset
    """
    sour = _get_file_path("uscities.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_tibetan() -> DataFrame:
    """load tibetan dataset
    Sophia Rabe-Hesketh i Brian Everitt, A Handbook of Statistical Analyses using Stata.
    Returns:
        pandas.DataFrame: tibetan dataset
    """
    sour = _get_file_path("tibetan.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_auto() -> DataFrame:
    """load auto dataset
    http://www.stata-press.com/data/r15/auto.dta
    Returns:
        pandas.DataFrame: auto dataset
    """
    sour = _get_file_path("auto.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_seul1988() -> DataFrame:
    """load seul1988 dataset
    Seul Olympics 1988, men decathlon results
    Returns:
        pandas.DataFrame: seul1988 dataset
    """
    sour = _get_file_path("seul1988.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_zadowolenie() -> DataFrame:
    """load zadowolenie dataset
    Returns:
        pandas.DataFrame: tibetan dataset
    """
    sour = _get_file_path("zadowolenie.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_euro() -> DataFrame:
    """load zadowolenie dataset
    Returns:
        pandas.DataFrame: euro dataset
    """
    sour = _get_file_path("euro.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_nauczyciele() -> DataFrame:
    """load zadowolenie dataset
    Returns:
        pandas.DataFrame: tibetan dataset
    """
    sour = _get_file_path("nauczyciele.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_depresja() -> DataFrame:
    """load depresja dataset
    Returns:
        pandas.DataFrame: depresja dataset
    """
    sour = _get_file_path("depresja.dta")
    with as_file(sour) as fil:
        return read_stata(fil)


def load_cerealnut() -> DataFrame:
    """load depresja dataset
    Returns:
        pandas.DataFrame: depresja dataset
    """
    sour = _get_file_path("cerealnut.dta")
    with as_file(sour) as fil:
        return read_stata(fil)
    
def load_boston() -> DataFrame:
    """load boston dataset
    Returns:
        pandas.DataFrame: depresja dataset
    """
    sour = _get_file_path("boston.dta")
    with as_file(sour) as fil:
        return read_stata(fil)
