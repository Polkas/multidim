from multidim.utils import resolve_stata, overwrite_stata_magic, load_stata
from unittest.mock import patch
from IPython.testing.globalipapp import get_ipython


def test_resolve_stata():
    current_stata = resolve_stata()
    assert isinstance(current_stata, tuple)
    assert len(current_stata) == 2


def test_overwrite_stata_magic():
    ip = get_ipython()
    assert overwrite_stata_magic() is None
    msg = "Stata was not loaded properly. Please check the STATA path and type."
    overwrite_stata_magic()
    assert ip.run_line_magic("stata", "display Hello") == msg
    assert ip.run_cell_magic("stata", "", "display Hello") == msg


def test_load_stata():
    load_stata(None, None)
