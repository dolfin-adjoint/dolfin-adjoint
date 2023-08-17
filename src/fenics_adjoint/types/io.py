import dolfin
from pyadjoint.tape import annotate_tape
from pyadjoint import OverloadedType

__all__ = []
__HDF5File_read__ = dolfin.HDF5File.read


def HDF5File_read(self, *args, **kwargs):
    annotate = annotate_tape(kwargs)
    output = __HDF5File_read__(self, *args, **kwargs)

    if annotate:
        func = args[0]
        if isinstance(func, OverloadedType):
            func.create_block_variable()
    return output


dolfin.HDF5File.read = HDF5File_read


__XDMFFile_read__ = dolfin.XDMFFile.read


def XDMFFile_read(self, *args, **kwargs):
    annotate = annotate_tape(kwargs)
    output = __XDMFFile_read__(self, *args, **kwargs)

    if annotate:
        func = args[0]
        if isinstance(func, OverloadedType):
            func.create_block_variable()
    return output


dolfin.XDMFFile.read = XDMFFile_read

__XDMFFile_read_checkpoint__ = dolfin.XDMFFile.read_checkpoint


def XDMFFile_read_checkpoint(self, *args, **kwargs):
    annotate = annotate_tape(kwargs)
    output = __XDMFFile_read_checkpoint__(self, *args, **kwargs)
    if annotate:
        func = args[0]
        if isinstance(func, OverloadedType):
            func.create_block_variable()
    return output


dolfin.XDMFFile.read_checkpoint = XDMFFile_read_checkpoint
