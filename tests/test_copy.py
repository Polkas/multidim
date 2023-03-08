from multidim import copy
import tempfile
import shutil
import pytest
from unittest.mock import patch
import os


def test_copy():
    with patch("sys.argv", ["copy", "WRONGPATH"]):
        with pytest.raises(TypeError):
            copy()
    dirpath = tempfile.mkdtemp()
    with patch("sys.argv", ["copy", dirpath]):
        copy()
        assert os.path.isdir(os.path.join(dirpath, "notebooks"))
        assert os.path.isfile(
            os.path.join(dirpath, "notebooks", "02_stats_intro.ipynb")
        )
    shutil.rmtree(dirpath)
