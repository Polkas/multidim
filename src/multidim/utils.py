from sys import platform
from os import path
from typing import Tuple
from collections import namedtuple
import warnings
from typing import Optional, Dict, List, NamedTuple, Callable
from IPython.core.magic import register_line_cell_magic


def resolve_stata(version: int = 17, stype: str = "se") -> NamedTuple:
    """Resolve the STATA version path and validate the type

    Args:
        version (int, optional): STATA version. Defaults to 17.
        stype (str, optional): STATA type has to be one of "se", "be", "mp". Defaults to "se".

    Returns:
        NamedTuple[str, str]: predicted STATA path and the validated STATA type
    >>> from multidim.utils import resolve_stata
    >>> resolve_stata()
    stata_setup(path=...
    """
    assert stype in ["se", "be", "mp"], 'stype has to be one of "se", "be", "mp"'
    assert isinstance(version, int), "version has to be an integer"

    stata_path: Optional[str] = None
    stata_path_assume: Optional[str] = None

    stata_paths_assume: Dict[str, List[str]] = {
        "linux": [f"/usr/local/stata{version}"],
        "linux2": [f"/usr/local/stata{version}"],
        "darwin": ["/Applications/Stata"],
        "win32": [
            f"C:\programy\Stata{version}",
            f"C:\Program Files\Stata{version}",
            f"C:\Program Files (x86)\Stata{version}",
            f"C:\Pliki Programów\Stata{version}",
            f"C:\Pliki Programów (x86)\Stata{version}",
        ],
    }

    assert platform in stata_paths_assume.keys(), "Platform not supported"

    for sp in stata_paths_assume[platform]:
        if path.exists(sp):
            stata_path_assume = sp
            break

    if stata_path_assume is None:
        warnings.warn("Automatic STATA path resolve FAILED.")
    else:
        stata_path = stata_path_assume

    stata_setup = namedtuple("stata_setup", ["path", "type"])
    return stata_setup(stata_path, stype)


def overwrite_stata_magic() -> None:
    """Overwrite the STATA magic to use the dummy one"""

    @register_line_cell_magic
    def stata(line: str, cell: Optional[str] = None) -> str:
        msg = "Stata was not loaded properly. Please check the STATA path and type."
        return msg
