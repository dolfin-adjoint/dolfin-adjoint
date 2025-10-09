import dolfin
from pyadjoint import Block
import numpy


class FunctionEvalBlock(Block):
    def __init__(self, func, coords, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.coords = coords

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        p = dolfin.Point(numpy.array(self.coords))
        V = inputs[0].function_space()
        dofs = V.dofmap()
        mesh = V.mesh()
        element = V.element()
        visited = []
        adj_vec = dolfin.Function(V).vector()
        adj_vec._function_space = V
        for cell_idx in range(len(mesh.cells())):
            cell = dolfin.Cell(mesh, cell_idx)
            if cell.contains(p):
                for ref_dof, dof in enumerate(dofs.cell_dofs(cell_idx)):
                    if dof in visited:
                        continue
                    visited.append(dof)
                    basis = element.evaluate_basis(ref_dof,
                                                   p.array(),
                                                   cell.get_coordinate_dofs(),
                                                   cell.orientation())
                    adj_vec[dof] = basis.dot(adj_inputs[idx])
        return adj_vec

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0](self.coords)


class FunctionSplitBlock(Block):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return dolfin.Function.sub(tlm_inputs[0], self.idx, deepcopy=False)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return dolfin.Function.sub(inputs[0], self.idx, deepcopy=False)


# TODO: This block is not valid in fenics and not correctly implemented. It should never be used.
class FunctionMergeBlock(Block):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        fs = output.output.function_space()
        f = dolfin.Function(fs)
        output.add_tlm_output(
            dolfin.assign(f.sub(self.idx), tlm_input)
        )

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute(self):
        dep = self.get_dependencies()[0].checkpoint
        output = self.get_outputs()[0].checkpoint
        dolfin.assign(dolfin.Function.sub(output, self.idx), dep)
