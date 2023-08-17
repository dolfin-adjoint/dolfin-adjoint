import dolfin

backend_fs_sub = dolfin.FunctionSpace.sub


def _fs_sub(self, i):
    V = backend_fs_sub(self, i)
    V._ad_parent_space = self
    return V


dolfin.FunctionSpace.sub = _fs_sub

dolfin.backend_fs_collapse = dolfin.FunctionSpace.collapse


def _fs_collapse(self, collapsed_dofs=False):
    """Overloaded FunctionSpace.collapse to limit the amount of MPI communicator created.
    """
    if not hasattr(self, "_ad_collapsed_space"):
        # Create collapsed space
        self._ad_collapsed_space = dolfin.backend_fs_collapse(self, collapsed_dofs=True)

    if collapsed_dofs:
        return self._ad_collapsed_space
    else:
        return self._ad_collapsed_space[0]


dolfin.FunctionSpace.collapse = _fs_collapse
