import pytest
pytest.importorskip("fenics")
from os import path
import subprocess

@pytest.mark.xfail(reason="Should work but testing interface is lacking for hessian with multiple controls")
def test(request):
    test_file = path.split(path.dirname(str(request.fspath)))[1] + ".py"
    test_dir = path.split(str(request.fspath))[0]
    test_cmd = ["python", path.join(test_dir, test_file)]

    handle = subprocess.Popen(test_cmd, cwd=test_dir)
    assert handle.wait() == 0
