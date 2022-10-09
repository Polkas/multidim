from sys import platform
from os import path
from typing import Tuple
from collections import namedtuple
import warnings
from typing import Optional


def resolve_stata(version: int = 17, stype: str = "se") -> Tuple[str]:
    """Resolve the STATA version path and validate the type

    Args:
        version (int, optional): STATA version. Defaults to 17.
        stype (str, optional): STATA type has to be one of "se", "be", "mp". Defaults to "se".

    Returns:
        NamedTuple[str, str]: predicted STATA path and the validated STATA type
    """
    assert stype in ["se", "be", "mp"], 'stype has to be one of "se", "be", "mp"'
    assert isinstance(version, int), "version has to be an integer"

    stata_path: Optional[str] = None
    stata_path_assume: Optional[str] = None

    if platform == "linux" or platform == "linux2":
        stata_path_assume = f"/usr/local/stata{version}"
    elif platform == "darwin":
        stata_path_assume = "/Applications/Stata"
    elif platform == "win32":
        stata_path_assume = f"C:\Program Files\Stata{version}"
        if not path.exists(stata_path_assume):
            stata_path_assume = f"C:\programy\Stata{version}"

    if path.exists(stata_path_assume):
        stata_path = stata_path_assume
    else:
        warnings.warn("Automatic STATA path resolve FAILED.")

    stata_setup = namedtuple("stata_setup", ["path", "type"])
    return stata_setup(stata_path, stype)
