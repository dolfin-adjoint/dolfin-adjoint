import dolfin


from pyadjoint.tape import stop_annotating

from fenics_adjoint.utils import MatrixTypes

_backend_SystemAssembler_assemble = dolfin.SystemAssembler.assemble
_backend_SystemAssembler_init = dolfin.SystemAssembler.__init__


def SystemAssembler_init(self, *args, **kwargs):
    _backend_SystemAssembler_init(self, *args, **kwargs)

    self._A_form = args[0]
    self._b_form = args[1]


def SystemAssembler_assemble(self, *args, **kwargs):
    with stop_annotating():
        out = _backend_SystemAssembler_assemble(self, *args, **kwargs)

    for arg in args:
        if isinstance(arg, dolfin.cpp.la.GenericVector):
            arg.form = self._b_form
            arg.bcs = self._bcs
        elif isinstance(arg, MatrixTypes):
            arg.form = self._A_form
            arg.bcs = self._bcs
            arg.assemble_system = True
        else:
            raise RuntimeError("Argument type not supported: ", type(arg))
    return out


dolfin.SystemAssembler.assemble = SystemAssembler_assemble
dolfin.SystemAssembler.__init__ = SystemAssembler_init
