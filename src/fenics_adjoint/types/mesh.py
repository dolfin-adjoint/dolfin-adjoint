import dolfin
import sys
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, FloatingType, register_overloaded_type
from ..shapead_transformations import vector_boundary_to_mesh, vector_mesh_to_boundary


overloaded_meshes = ['IntervalMesh', 'UnitIntervalMesh', 'RectangleMesh',
                     'UnitSquareMesh', 'UnitCubeMesh', 'BoxMesh']
__all__ = ['Mesh', 'BoundaryMesh', 'SubMesh'] + overloaded_meshes


@register_overloaded_type
class Mesh(OverloadedType, dolfin.Mesh):
    def __init__(self, *args, **kwargs):
        # Calling constructor
        super(Mesh, self).__init__(*args, **kwargs)
        dolfin.Mesh.__init__(self, *args, **kwargs)

        self._ad_coordinate_space = None

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self

    def _ad_function_space(self):
        if self._ad_coordinate_space is None:
            self._ad_coordinate_space = dolfin.FunctionSpace(self, self.ufl_coordinate_element())
        return self._ad_coordinate_space


@register_overloaded_type
class BoundaryMesh(FloatingType, dolfin.BoundaryMesh):
    def __init__(self, *args, **kwargs):
        # Calling constructor
        super(BoundaryMesh, self).__init__(*args,
                                           block_class=BoundaryMeshBlock,
                                           _ad_args=args,
                                           _ad_floating_active=True,
                                           annotate=kwargs.pop("annotate", True),
                                           **kwargs)
        dolfin.BoundaryMesh.__init__(self, *args, **kwargs)
        self._ad_coordinate_space = None

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self

    def _ad_function_space(self):
        if self._ad_coordinate_space is None:
            self._ad_coordinate_space = dolfin.FunctionSpace(self, self.ufl_coordinate_element())
        return self._ad_coordinate_space


def overloaded_mesh(mesh_class):
    @register_overloaded_type
    class OverloadedMesh(OverloadedType, mesh_class):
        def __init__(self, *args, **kwargs):
            # Calling constructor
            super(OverloadedMesh, self).__init__(*args, **kwargs)
            mesh_class.__init__(self, *args, **kwargs)
            self._ad_coordinate_space = None

        def _ad_create_checkpoint(self):
            return self.coordinates().copy()

        def _ad_restore_at_checkpoint(self, checkpoint):
            self.coordinates()[:] = checkpoint
            return self

        def _ad_function_space(self):
            if self._ad_coordinate_space is None:
                self._ad_coordinate_space = dolfin.FunctionSpace(self, self.ufl_coordinate_element())
            return self._ad_coordinate_space

    return OverloadedMesh


thismod = sys.modules[__name__]
for name in overloaded_meshes:
    setattr(thismod, name, overloaded_mesh(getattr(dolfin, name)))
    mod = getattr(thismod, name)
    mod.__doc__ = getattr(dolfin, name).init.__doc__

SubMesh = overloaded_mesh(dolfin.SubMesh)
SubMesh.__doc__ = dolfin.SubMesh.__doc__


def overloaded_create(mesh_class):
    __ad_create = mesh_class.create

    def create(*args, **kwargs):
        mesh = __ad_create(*args, **kwargs)
        mesh.__class__ = Mesh
        mesh.__init__()

        return mesh
    create.__doc__ = __ad_create.__doc__
    return create


custom_meshes = ["UnitDiscMesh", "SphericalShellMesh", "UnitTriangleMesh", "BoxMesh", "RectangleMesh"]
for name in custom_meshes:
    mesh_type = getattr(dolfin, name)
    mesh_type.create = overloaded_create(mesh_type)


__backend_ALE_move = dolfin.ALE.move


def move(mesh, vector, **kwargs):
    annotate = annotate_tape(kwargs)
    if annotate:
        assert isinstance(mesh, OverloadedType)
        assert isinstance(vector, OverloadedType)
        tape = get_working_tape()
        block = ALEMoveBlock(mesh, vector, **kwargs)
        tape.add_block(block)

    with stop_annotating():
        output = __backend_ALE_move(mesh, vector)
    if annotate:
        block.add_output(mesh.create_block_variable())
    return output


dolfin.ALE.move = move


class ALEMoveBlock(Block):
    def __init__(self, mesh, vector, **kwargs):
        super(ALEMoveBlock, self).__init__(**kwargs)
        self.add_dependency(mesh)
        self.add_dependency(vector)

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return
        # NOTE: Hack to pass FunctionSpaces with Vectors
        adj_value._function_space = self.get_dependencies()[1].checkpoint.function_space()
        self.get_dependencies()[0].add_adj_output(adj_value.copy())
        self.get_dependencies()[1].add_adj_output(adj_value)

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_output = None

        for i in range(2):
            tlm_input = self.get_dependencies()[i].tlm_value
            if tlm_input is None:
                continue
            if tlm_output is None:
                tlm_output = tlm_input.copy(deepcopy=True)
            else:
                tlm_output.vector().axpy(1, tlm_input.vector())

        if tlm_output is not None:
            self.get_outputs()[0].add_tlm_output(tlm_output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        hessian_input._function_space = self.get_dependencies()[1].checkpoint.function_space()

        self.get_dependencies()[0].add_hessian_output(hessian_input.copy())
        self.get_dependencies()[1].add_hessian_output(hessian_input)

    @no_annotations
    def recompute(self, markings=False):
        mesh = self.get_dependencies()[0].saved_output
        vector = self.get_dependencies()[1].saved_output

        dolfin.ALE.move(mesh, vector, annotate=False)

        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()


class BoundaryMeshBlock(Block):
    def __init__(self, mesh, mesh_type, **kwargs):
        super(BoundaryMeshBlock, self).__init__(**kwargs)
        self.add_dependency(mesh)
        self.mesh_type = mesh_type

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        f = dolfin.Function(dolfin.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
        f.vector()[:] = adj_value
        adj_value = vector_boundary_to_mesh(f, self.get_dependencies()[0].saved_output)
        vec = adj_value.vector()
        vec._function_space = adj_value.function_space()
        self.get_dependencies()[0].add_adj_output(vec)

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return

        tlm_output = vector_mesh_to_boundary(tlm_input, self.get_outputs()[0].saved_output)
        self.get_outputs()[0].add_tlm_output(tlm_output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return

        f = dolfin.Function(dolfin.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
        f.vector()[:] = hessian_input
        hessian_value = vector_boundary_to_mesh(f, self.get_dependencies()[0].saved_output)
        vec = hessian_value.vector()
        vec._function_space = hessian_value.function_space()
        self.get_dependencies()[0].add_hessian_output(vec)

    @no_annotations
    def recompute(self):
        mesh = self.get_dependencies()[0].saved_output

        b_mesh = BoundaryMesh(mesh, self.mesh_type, annotate=False)

        self.get_outputs()[0].checkpoint = b_mesh._ad_create_checkpoint()
