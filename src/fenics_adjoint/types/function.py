import dolfin
import numpy
import ufl_legacy as ufl
from pyadjoint import AdjFloat, Block, OverloadedType
from pyadjoint.enlisting import Enlist
from pyadjoint.overloaded_type import (FloatingType, create_overloaded_object,
                                       get_overloaded_class,
                                       register_overloaded_type)
from pyadjoint.tape import (annotate_tape, get_working_tape, no_annotations,
                            stop_annotating)
from ufl_legacy.corealg.traversal import traverse_unique_terminals
from ufl_legacy.formatting.ufl2unicode import ufl2unicode

from fenics_adjoint.blocks import (FunctionEvalBlock, FunctionMergeBlock,
                                   FunctionSplitBlock)
from fenics_adjoint.utils import function_from_vector, gather, linalg_solve, create_function
from .constant import create_constant


def type_cast_function(obj, cls):
    """Type casts Function object `obj` to an instance of `cls`.

    Useful when converting backend.Function to overloaded Function.
    """
    return cls(obj.function_space(), obj._cpp_object)


@register_overloaded_type
class Function(FloatingType, dolfin.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args,
                                       block_class=kwargs.pop("block_class",
                                                              None),
                                       _ad_floating_active=kwargs.pop(
                                           "_ad_floating_active", False),
                                       _ad_args=kwargs.pop("_ad_args", None),
                                       output_block_class=kwargs.pop(
                                           "output_block_class", None),
                                       _ad_output_args=kwargs.pop(
                                           "_ad_output_args", None),
                                       _ad_outputs=kwargs.pop("_ad_outputs",
                                                              None),
                                       annotate=kwargs.pop("annotate", True),
                                       **kwargs)
        dolfin.Function.__init__(self, *args, **kwargs)

    @classmethod
    def _ad_init_object(cls, obj):
        return type_cast_function(obj, cls)

    def copy(self, *args, **kwargs):
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        c = dolfin.Function.copy(self, *args, **kwargs)
        func = create_overloaded_object(c)

        if annotate:
            if kwargs.pop("deepcopy", False):
                block = FunctionAssignBlock(func, self, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())
            else:
                # TODO: Implement. Here we would need to use floating types.
                pass

        return func

    def assign(self, other, *args, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin assign call."""
        # do not annotate in case of self assignment
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs) and self != other
        if annotate:
            if not isinstance(other, ufl.core.operator.Operator):
                other = create_overloaded_object(other)
            block = FunctionAssignBlock(self, other, ad_block_tag=ad_block_tag)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            ret = super(Function, self).assign(other, *args, **kwargs)

        if annotate:
            block.add_output(self.create_block_variable())

        return ret

    def sub(self, i, deepcopy=False, **kwargs):
        from .function_assigner import FunctionAssigner, FunctionAssignerBlock
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        if deepcopy:
            ret = create_overloaded_object(dolfin.Function.sub(self, i, deepcopy, **kwargs))
            if annotate:
                fa = FunctionAssigner(ret.function_space(), self.function_space())
                block = FunctionAssignerBlock(fa, Enlist(self), ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(ret.block_variable)
        else:
            extra_kwargs = {}
            if annotate:
                extra_kwargs = {
                    "block_class": FunctionSplitBlock,
                    "_ad_floating_active": True,
                    "_ad_args": [self, i],
                    "_ad_output_args": [i],
                    "output_block_class": FunctionMergeBlock,
                    "_ad_outputs": [self],
                }
            ret = create_function(self, i, **extra_kwargs)
        return ret

    def split(self, deepcopy=False, **kwargs):
        from .function_assigner import FunctionAssigner, FunctionAssignerBlock
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        num_sub_spaces = dolfin.Function.function_space(self).num_sub_spaces()
        if not annotate:
            if deepcopy:
                ret = tuple(create_overloaded_object(dolfin.Function.sub(self, i, deepcopy, **kwargs))
                            for i in range(num_sub_spaces))
            else:
                ret = tuple(create_function(self, i)
                            for i in range(num_sub_spaces))
        elif deepcopy:
            ret = []
            fs = []
            for i in range(num_sub_spaces):

                f = create_overloaded_object(dolfin.Function.sub(self, i, deepcopy, **kwargs))
                fs.append(f.function_space())
                ret.append(f)
            fa = FunctionAssigner(fs, self.function_space())
            block = FunctionAssignerBlock(fa, Enlist(self), ad_block_tag=ad_block_tag)
            tape = get_working_tape()
            tape.add_block(block)
            for output in ret:
                block.add_output(output.block_variable)
            ret = tuple(ret)
        else:
            ret = tuple(Function(self,
                                 i,
                                 block_class=FunctionSplitBlock,
                                 _ad_floating_active=True,
                                 _ad_args=[self, i],
                                 _ad_output_args=[i],
                                 output_block_class=FunctionMergeBlock,
                                 _ad_outputs=[self])
                        for i in range(num_sub_spaces))
        return ret

    def __call__(self, *args, **kwargs):
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = False
        if len(args) == 1 and isinstance(args[0], (numpy.ndarray,)):
            annotate = annotate_tape(kwargs)

        if annotate:
            block = FunctionEvalBlock(self, args[0], ad_block_tag=ad_block_tag)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = dolfin.Function.__call__(self, *args, **kwargs)

        if annotate:
            out = create_overloaded_object(out)
            block.add_output(out.create_block_variable())

        return out

    def vector(self):
        vec = dolfin.Function.vector(self)
        vec.function = self
        return vec

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")

        if riesz_representation == "l2":
            return create_overloaded_object(
                function_from_vector(self.function_space(), value)
            )
        elif riesz_representation == "L2":
            ret = Function(self.function_space())
            u = dolfin.TrialFunction(self.function_space())
            v = dolfin.TestFunction(self.function_space())
            M = dolfin.assemble(dolfin.inner(u, v) * dolfin.dx)
            linalg_solve(M, ret.vector(), value)
            return ret
        elif riesz_representation == "H1":
            ret = Function(self.function_space())
            u = dolfin.TrialFunction(self.function_space())
            v = dolfin.TestFunction(self.function_space())
            M = dolfin.assemble(
                dolfin.inner(u, v) * dolfin.dx + dolfin.inner(
                    dolfin.grad(u), dolfin.grad(v)) * dolfin.dx)
            linalg_solve(M, ret.vector(), value)
            return ret
        elif callable(riesz_representation):
            return riesz_representation(value)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @no_annotations
    def _ad_create_checkpoint(self):
        if self.block is None:
            # TODO: This might crash if annotate=False, but still using a sub-function.
            #       Because subfunction.copy(deepcopy=True) raises the can't access vector error.
            return self.copy(deepcopy=True)

        dep = self.block.get_dependencies()[0].saved_output
        return dep.sub(self.block.idx, deepcopy=False)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def _ad_mul(self, other):
        r = get_overloaded_class(dolfin.Function)(self.function_space())
        dolfin.Function.assign(r, self * other)
        return r

    @no_annotations
    def _ad_add(self, other):
        r = get_overloaded_class(dolfin.Function)(self.function_space())
        dolfin.Function.assign(r, self + other)
        return r

    def _ad_dot(self, other, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return self.vector().inner(other.vector())
        elif riesz_representation == "L2":
            return dolfin.assemble(dolfin.inner(self, other) * dolfin.dx)
        elif riesz_representation == "H1":
            return dolfin.assemble(
                (dolfin.inner(self, other) + dolfin.inner(dolfin.grad(self),
                                                          dolfin.grad(
                    other))) * dolfin.dx)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        range_begin, range_end = dst.vector().local_range()
        m_a_local = src[offset + range_begin:offset + range_end]
        dst.vector().set_local(m_a_local)
        dst.vector().apply('insert')
        offset += dst.vector().size()
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        if not hasattr(m, "gather"):
            m_v = m.vector()
        else:
            m_v = m
        m_a = gather(m_v)

        return m_a.tolist()

    def _ad_copy(self):
        r = get_overloaded_class(dolfin.Function)(self.function_space())
        dolfin.Function.assign(r, self)
        return r

    def _ad_dim(self):
        return self.function_space().dim()

    def _ad_imul(self, other):
        vec = self.vector()
        vec *= other

    def _ad_iadd(self, other):
        vec = self.vector()
        # FIXME: PETSc complains when we add the same vector to itself.
        # So we make a copy.
        vec += other.vector().copy()

    def _reduce(self, r, r0):
        vec = self.vector().get_local()
        for i in range(len(vec)):
            r0 = r(vec[i], r0)
        return r0

    def _applyUnary(self, f):
        vec = self.vector()
        npdata = vec.get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i])
        vec.set_local(npdata)
        vec.apply("insert")

    def _applyBinary(self, f, y):
        vec = self.vector()
        npdata = vec.get_local()
        npdatay = y.vector().get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i], npdatay[i])
        vec.set_local(npdata)
        vec.apply("insert")

    def __deepcopy__(self, memodict={}):
        return self.copy(deepcopy=True)


class FunctionAssignBlock(Block):
    def __init__(self, func, other, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.other = None
        self.expr = None
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif isinstance(other, float) or isinstance(other, int):
            other = AdjFloat(other)
            self.add_dependency(other, no_duplicates=True)
        elif not (isinstance(other, float) or isinstance(other, int)):
            # Assume that this is a point-wise evaluated UFL expression (firedrake only)
            for op in traverse_unique_terminals(other):
                if isinstance(op, OverloadedType):
                    self.add_dependency(op, no_duplicates=True)
            self.expr = other

    def _replace_with_saved_output(self):
        if self.expr is None:
            return None

        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        return ufl.replace(self.expr, replace_map)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        V = self.get_outputs()[0].output.function_space()
        adj_input_func = function_from_vector(V, adj_inputs[0])

        if self.expr is None:
            return adj_input_func

        expr = self._replace_with_saved_output()
        return expr, adj_input_func

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            if isinstance(block_variable.output, AdjFloat):
                try:
                    # Adjoint of a broadcast is just a sum
                    return adj_inputs[0].sum()
                except AttributeError:
                    # Catch the case where adj_inputs[0] is just a float
                    return adj_inputs[0]
            elif isinstance(block_variable.output, dolfin.Constant):
                R = block_variable.output._ad_function_space(prepared.function_space().mesh())
                return self._adj_assign_constant(prepared, R)
            else:
                adj_output = dolfin.Function(
                    block_variable.output.function_space())
                adj_output.assign(prepared)
                return adj_output.vector()
        else:
            # Linear combination
            expr, adj_input_func = prepared
            adj_output = dolfin.Function(adj_input_func.function_space())
            if not isinstance(block_variable.output, dolfin.Constant):
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, adj_input_func)
                )
                adj_output.assign(diff_expr)
            else:
                mesh = adj_output.function_space().mesh()
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        expr,
                        block_variable.saved_output,
                        create_constant(1., domain=mesh)
                    )
                )
                adj_output.assign(diff_expr)
                return adj_output.vector().inner(adj_input_func.vector())

            if isinstance(block_variable.output, dolfin.Constant):
                R = block_variable.output._ad_function_space(adj_output.function_space().mesh())
                return self._adj_assign_constant(adj_output, R)
            else:
                return adj_output.vector()

    def _adj_assign_constant(self, adj_output, constant_fs):
        r = dolfin.Function(constant_fs)
        shape = r.ufl_shape
        if shape == () or shape[0] == 1:
            # Scalar Constant
            r.vector()[:] = adj_output.vector().sum()
        else:
            # We assume the shape of the constant == shape of the output function if not scalar.
            # This assumption is due to FEniCS not supporting products with non-scalar constants in assign.
            values = []
            for i in range(shape[0]):
                values.append(adj_output.sub(i, deepcopy=True).vector().sum())
            r.assign(dolfin.Constant(values))
        return r.vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            return tlm_inputs[0]

        expr = prepared
        dudm = dolfin.Function(block_variable.output.function_space())
        dudmi = dolfin.Function(block_variable.output.function_space())
        for dep in self.get_dependencies():
            if dep.tlm_value:
                dudmi.assign(ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, dep.saved_output,
                                   dep.tlm_value)))
                dudm.vector().axpy(1.0, dudmi.vector())

        return dudm

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        # Current implementation assumes lincom in hessian,
        # otherwise we need second-order derivatives here.
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx, prepared)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        if self.expr is None:
            return None
        return self._replace_with_saved_output()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if self.expr is None:
            prepared = inputs[0]
        output = dolfin.Function(block_variable.output.function_space())
        output.assign(prepared)
        return output

    def __str__(self):
        rhs = self.expr or self.other or self.get_dependencies()[0].output
        if isinstance(rhs, ufl.core.expr.Expr):
            rhs_str = ufl2unicode(rhs)
        else:
            rhs_str = str(rhs)
        return f"assign({rhs_str})"
