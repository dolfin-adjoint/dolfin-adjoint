import pytest
pytest.importorskip("fenics")
import pytest
from os import path
import subprocess

@pytest.mark.skipif_module_is_missing("petsc4py")
@pytest.mark.xfail(reason="PETScKrylovSolver is not implemented")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = ["python", path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
