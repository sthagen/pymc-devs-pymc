#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from typing import List, Optional, Tuple, Union, cast

import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op, compute_test_value
from pytensor.graph.rewriting.basic import node_rewriter, pre_greedy_node_rewriter
from pytensor.ifelse import IfElse, ifelse
from pytensor.tensor.basic import Join, MakeVector, switch
from pytensor.tensor.random.rewriting import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from pytensor.tensor.shape import shape_tuple
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    as_index_literal,
    as_nontensor_scalar,
    get_canonical_form_slice,
    is_basic_idx,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneConst, NoneTypeT, SliceConstant, SliceType
from pytensor.tensor.var import TensorVariable

from pymc.logprob.abstract import MeasurableVariable, _logprob, _logprob_helper
from pymc.logprob.rewriting import (
    PreserveRVMappings,
    assume_measured_ir_outputs,
    local_lift_DiracDelta,
    measurable_ir_rewrites_db,
    subtensor_ops,
)
from pymc.logprob.tensor import naive_bcast_rv_lift
from pymc.logprob.utils import check_potential_measurability


def is_newaxis(x):
    return isinstance(x, type(None)) or isinstance(getattr(x, "type", None), NoneTypeT)


def expand_indices(
    indices: Tuple[Optional[Union[Variable, slice]], ...], shape: Tuple[TensorVariable]
) -> Tuple[TensorVariable]:
    """Convert basic and/or advanced indices into a single, broadcasted advanced indexing operation.

    Parameters
    ----------
    indices
        The indices to convert.
    shape
        The shape of the array being indexed.

    """
    n_non_newaxis = sum(1 for idx in indices if not is_newaxis(idx))
    n_missing_dims = len(shape) - n_non_newaxis
    full_indices = list(indices) + [slice(None)] * n_missing_dims

    # We need to know if a "subspace" was generated by advanced indices
    # bookending basic indices.  If so, we move the advanced indexing subspace
    # to the "front" of the shape (i.e. left-most indices/last-most
    # dimensions).
    index_types = [is_basic_idx(idx) for idx in full_indices]

    first_adv_idx = len(shape)
    try:
        first_adv_idx = index_types.index(False)
        first_bsc_after_adv_idx = index_types.index(True, first_adv_idx)
        index_types.index(False, first_bsc_after_adv_idx)
        moved_subspace = True
    except ValueError:
        moved_subspace = False

    n_basic_indices = sum(index_types)

    # The number of dimensions in the subspace created by the advanced indices
    n_subspace_dims = max(
        (
            getattr(idx, "ndim", 0)
            for idx, is_basic in zip(full_indices, index_types)
            if not is_basic
        ),
        default=0,
    )

    # The number of dimensions for each expanded index
    n_output_dims = n_subspace_dims + n_basic_indices

    adv_indices = []
    shape_copy = list(shape)
    n_preceding_basics = 0
    for d, idx in enumerate(full_indices):
        if not is_basic_idx(idx):
            s = shape_copy.pop(0)

            idx = pt.as_tensor(idx)

            if moved_subspace:
                # The subspace generated by advanced indices appear as the
                # upper dimensions in the "expanded" index space, so we need to
                # add broadcast dimensions for the non-basic indices to the end
                # of these advanced indices
                expanded_idx = idx[(Ellipsis,) + (None,) * n_basic_indices]
            else:
                # In this case, we need to add broadcast dimensions for the
                # basic indices that proceed and follow the group of advanced
                # indices; otherwise, a contiguous group of advanced indices
                # forms a broadcasted set of indices that are iterated over
                # within the same subspace, which means that all their
                # corresponding "expanded" indices have exactly the same shape.
                expanded_idx = idx[(None,) * n_preceding_basics][
                    (Ellipsis,) + (None,) * (n_basic_indices - n_preceding_basics)
                ]
        else:
            if is_newaxis(idx):
                n_preceding_basics += 1
                continue

            s = shape_copy.pop(0)

            if isinstance(idx, slice) or isinstance(getattr(idx, "type", None), SliceType):
                idx = as_index_literal(idx)
                idx_slice, _ = get_canonical_form_slice(idx, s)
                idx = pt.arange(idx_slice.start, idx_slice.stop, idx_slice.step)

            if moved_subspace:
                # Basic indices appear in the lower dimensions
                # (i.e. right-most) in the output, and are preceded by
                # the subspace generated by the advanced indices.
                expanded_idx = idx[(None,) * (n_subspace_dims + n_preceding_basics)][
                    (Ellipsis,) + (None,) * (n_basic_indices - n_preceding_basics - 1)
                ]
            else:
                # In this case, we need to know when the basic indices have
                # moved past the contiguous group of advanced indices (in the
                # "expanded" index space), so that we can properly pad those
                # dimensions in this basic index's shape.
                # Don't forget that a single advanced index can introduce an
                # arbitrary number of dimensions to the expanded index space.

                # If we're currently at a basic index that's past the first
                # advanced index, then we're necessarily past the group of
                # advanced indices.
                n_preceding_dims = (
                    n_subspace_dims if d > first_adv_idx else 0
                ) + n_preceding_basics
                expanded_idx = idx[(None,) * n_preceding_dims][
                    (Ellipsis,) + (None,) * (n_output_dims - n_preceding_dims - 1)
                ]

            n_preceding_basics += 1

        assert expanded_idx.ndim <= n_output_dims

        adv_indices.append(expanded_idx)

    return cast(Tuple[TensorVariable], tuple(pt.broadcast_arrays(*adv_indices)))


def rv_pull_down(x: TensorVariable, dont_touch_vars=None) -> TensorVariable:
    """Pull a ``RandomVariable`` ``Op`` down through a graph, when possible."""
    fgraph = FunctionGraph(outputs=dont_touch_vars or [], clone=False)

    return pre_greedy_node_rewriter(
        fgraph,
        [
            local_rv_size_lift,
            local_dimshuffle_rv_lift,
            local_subtensor_rv_lift,
            naive_bcast_rv_lift,
            local_lift_DiracDelta,
        ],
        x,
    )


class MixtureRV(Op):
    """A placeholder used to specify a log-likelihood for a mixture sub-graph."""

    __props__ = ("indices_end_idx", "out_dtype", "out_broadcastable")

    def __init__(self, indices_end_idx, out_dtype, out_broadcastable):
        super().__init__()
        self.indices_end_idx = indices_end_idx
        self.out_dtype = out_dtype
        self.out_broadcastable = out_broadcastable

    def make_node(self, *inputs):
        return Apply(self, list(inputs), [TensorType(self.out_dtype, self.out_broadcastable)()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("This is a stand-in Op.")  # pragma: no cover


MeasurableVariable.register(MixtureRV)


def get_stack_mixture_vars(
    node: Apply,
) -> Tuple[Optional[List[TensorVariable]], Optional[int]]:
    r"""Extract the mixture terms from a `*Subtensor*` applied to stacked `MeasurableVariable`\s."""

    assert isinstance(node.op, subtensor_ops)

    joined_rvs = node.inputs[0]

    # First, make sure that it's some sort of concatenation
    if not (joined_rvs.owner and isinstance(joined_rvs.owner.op, (MakeVector, Join))):
        return None, None

    if isinstance(joined_rvs.owner.op, MakeVector):
        join_axis = NoneConst
        mixture_rvs = joined_rvs.owner.inputs

    elif isinstance(joined_rvs.owner.op, Join):
        # TODO: Find better solution to avoid this circular dependency
        from pymc.pytensorf import constant_fold

        join_axis = joined_rvs.owner.inputs[0]
        # TODO: Support symbolic join axes. This will raise ValueError if it's not a constant
        (join_axis,) = constant_fold((join_axis,), raise_not_constant=False)
        join_axis = pt.as_tensor(join_axis, dtype="int64")

        mixture_rvs = joined_rvs.owner.inputs[1:]

    return mixture_rvs, join_axis


@node_rewriter(subtensor_ops)
def find_measurable_index_mixture(fgraph, node):
    r"""Identify mixture sub-graphs and replace them with a place-holder `Op`.

    The basic idea is to find ``stack(mixture_comps)[I_rv]``, where
    ``mixture_comps`` is a ``list`` of `MeasurableVariable`\s and ``I_rv`` is a
    `MeasurableVariable` with a discrete and finite support.
    From these terms, new terms ``Z_rv[i] = mixture_comps[i][i == I_rv]`` are
    created for each ``i`` in ``enumerate(mixture_comps)``.
    """
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    mixing_indices = node.inputs[1:]

    # TODO: Add check / test case for Advanced Boolean indexing
    if isinstance(node.op, (AdvancedSubtensor, AdvancedSubtensor1)):
        # We don't support (non-scalar) integer array indexing as it can pick repeated values,
        # but the Mixture logprob assumes all mixture values are independent
        if any(
            indices.dtype.startswith("int") and sum(1 - b for b in indices.type.broadcastable) > 0
            for indices in mixing_indices
            if not isinstance(indices, SliceConstant)
        ):
            return None

    old_mixture_rv = node.default_output()
    mixture_rvs, join_axis = get_stack_mixture_vars(node)

    # We don't support symbolic join axis
    if mixture_rvs is None or not isinstance(join_axis, (NoneTypeT, Constant)):
        return None

    if rv_map_feature.request_measurable(mixture_rvs) != mixture_rvs:
        return None

    # Replace this sub-graph with a `MixtureRV`
    mix_op = MixtureRV(
        1 + len(mixing_indices),
        old_mixture_rv.dtype,
        old_mixture_rv.broadcastable,
    )
    new_node = mix_op.make_node(*([join_axis] + mixing_indices + mixture_rvs))

    new_mixture_rv = new_node.default_output()

    if pytensor.config.compute_test_value != "off":
        # We can't use `MixtureRV` to compute a test value; instead, we'll use
        # the original node's test value.
        if not hasattr(old_mixture_rv.tag, "test_value"):
            compute_test_value(node)

        new_mixture_rv.tag.test_value = old_mixture_rv.tag.test_value

    return [new_mixture_rv]


@node_rewriter([switch])
def find_measurable_switch_mixture(fgraph, node):
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    old_mixture_rv = node.default_output()
    idx, *components = node.inputs

    if rv_map_feature.request_measurable(components) != components:
        return None

    mix_op = MixtureRV(
        2,
        old_mixture_rv.dtype,
        old_mixture_rv.broadcastable,
    )
    new_mixture_rv = mix_op.make_node(
        *([NoneConst, as_nontensor_scalar(node.inputs[0])] + components[::-1])
    ).default_output()

    if pytensor.config.compute_test_value != "off":
        if not hasattr(old_mixture_rv.tag, "test_value"):
            compute_test_value(node)

        new_mixture_rv.tag.test_value = old_mixture_rv.tag.test_value

    return [new_mixture_rv]


@_logprob.register(MixtureRV)
def logprob_MixtureRV(
    op, values, *inputs: Optional[Union[TensorVariable, slice]], name=None, **kwargs
):
    (value,) = values

    join_axis = cast(Variable, inputs[0])
    indices = cast(TensorVariable, inputs[1 : op.indices_end_idx])
    comp_rvs = cast(TensorVariable, inputs[op.indices_end_idx :])

    assert len(indices) > 0

    if len(indices) > 1 or indices[0].ndim > 0:
        if isinstance(join_axis.type, NoneTypeT):
            # `join_axis` will be `NoneConst` if the "join" was a `MakeVector`
            # (i.e. scalar measurable variables were combined to make a
            # vector).
            # Since some form of advanced indexing is necessarily occurring, we
            # need to reformat the MakeVector arguments so that they fit the
            # `Join` format expected by the logic below.
            join_axis_val = 0
            comp_rvs = [comp[None] for comp in comp_rvs]
            original_shape = (len(comp_rvs),)
        else:
            # TODO: Find better solution to avoid this circular dependency
            from pymc.pytensorf import constant_fold

            join_axis_val = constant_fold((join_axis,))[0].item()
            original_shape = shape_tuple(comp_rvs[0])

        bcast_indices = expand_indices(indices, original_shape)

        logp_val = pt.empty(bcast_indices[0].shape)

        for m, rv in enumerate(comp_rvs):
            idx_m_on_axis = pt.nonzero(pt.eq(bcast_indices[join_axis_val], m))
            m_indices = tuple(
                v[idx_m_on_axis] for i, v in enumerate(bcast_indices) if i != join_axis_val
            )
            # Drop superfluous join dimension
            rv = rv[0]
            # TODO: Do we really need to do this now?
            # Could we construct this form earlier and
            # do the lifting for everything at once, instead of
            # this intentional one-off?
            rv_m = rv_pull_down(rv[m_indices] if m_indices else rv)
            val_m = value[idx_m_on_axis]
            logp_m = _logprob_helper(rv_m, val_m)
            logp_val = pt.set_subtensor(logp_val[idx_m_on_axis], logp_m)

    else:
        # FIXME: This logprob implementation does not support mixing across distinct components,
        # but we sometimes use it, because MixtureRV does not keep information about at which
        # dimension scalar indexing actually starts

        # If the stacking operation expands the component RVs, we have
        # to expand the value and later squeeze the logprob for everything
        # to work correctly
        join_axis_val = None if isinstance(join_axis.type, NoneTypeT) else join_axis.data

        if join_axis_val is not None:
            value = pt.expand_dims(value, axis=join_axis_val)

        logp_val = 0.0
        for i, comp_rv in enumerate(comp_rvs):
            comp_logp = _logprob_helper(comp_rv, value)
            if join_axis_val is not None:
                comp_logp = pt.squeeze(comp_logp, axis=join_axis_val)
            logp_val += ifelse(
                pt.eq(indices[0], i),
                comp_logp,
                pt.zeros_like(comp_logp),
            )

    return logp_val


measurable_ir_rewrites_db.register(
    "find_measurable_index_mixture",
    find_measurable_index_mixture,
    "basic",
    "mixture",
)

measurable_ir_rewrites_db.register(
    "find_measurable_switch_mixture",
    find_measurable_switch_mixture,
    "basic",
    "mixture",
)


class MeasurableIfElse(IfElse):
    """Measurable subclass of IfElse operator."""


MeasurableVariable.register(MeasurableIfElse)


@node_rewriter([IfElse])
def useless_ifelse_outputs(fgraph, node):
    """Remove outputs that are shared across the IfElse branches."""
    # TODO: This should be a PyTensor canonicalization
    op = node.op
    if_var, *inputs = node.inputs
    shared_inputs = set(inputs[op.n_outs :]).intersection(inputs[: op.n_outs])
    if not shared_inputs:
        return None

    replacements = {}
    for shared_inp in shared_inputs:
        idx = inputs.index(shared_inp)
        replacements[node.outputs[idx]] = shared_inp

    # IfElse isn't needed at all
    if len(shared_inputs) == op.n_outs:
        return replacements

    # Create subset IfElse with remaining nodes
    remaining_inputs = [inp for inp in inputs if inp not in shared_inputs]
    new_outs = (
        IfElse(n_outs=len(remaining_inputs) // 2).make_node(if_var, *remaining_inputs).outputs
    )
    for inp, new_out in zip(remaining_inputs, new_outs):
        idx = inputs.index(inp)
        replacements[node.outputs[idx]] = new_out

    return replacements


@node_rewriter([IfElse])
def find_measurable_ifelse_mixture(fgraph, node):
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    op = node.op
    if_var, *base_rvs = node.inputs

    valued_rvs = rv_map_feature.rv_values.keys()
    if not all(check_potential_measurability([base_var], valued_rvs) for base_var in base_rvs):
        return None

    base_rvs = assume_measured_ir_outputs(valued_rvs, base_rvs)
    if len(base_rvs) != op.n_outs * 2:
        return None
    if not all(var.owner and isinstance(var.owner.op, MeasurableVariable) for var in base_rvs):
        return None

    return MeasurableIfElse(n_outs=op.n_outs).make_node(if_var, *base_rvs).outputs


measurable_ir_rewrites_db.register(
    "useless_ifelse_outputs",
    useless_ifelse_outputs,
    "basic",
    "mixture",
)


measurable_ir_rewrites_db.register(
    "find_measurable_ifelse_mixture",
    find_measurable_ifelse_mixture,
    "basic",
    "mixture",
)


@_logprob.register(MeasurableIfElse)
def logprob_ifelse(op, values, if_var, *base_rvs, **kwargs):
    """Compute the log-likelihood graph for an `IfElse`."""
    from pymc.pytensorf import replace_rvs_by_values

    assert len(values) * 2 == len(base_rvs)

    rvs_to_values_then = {then_rv: value for then_rv, value in zip(base_rvs[: len(values)], values)}
    rvs_to_values_else = {else_rv: value for else_rv, value in zip(base_rvs[len(values) :], values)}

    logps_then = [
        _logprob_helper(rv_then, value, **kwargs) for rv_then, value in rvs_to_values_then.items()
    ]
    logps_else = [
        _logprob_helper(rv_else, value, **kwargs) for rv_else, value in rvs_to_values_else.items()
    ]

    # If the multiple variables depend on each other, we have to replace them
    # by the respective values
    logps_then = replace_rvs_by_values(logps_then, rvs_to_values=rvs_to_values_then)
    logps_else = replace_rvs_by_values(logps_else, rvs_to_values=rvs_to_values_else)

    logps = ifelse(if_var, logps_then, logps_else)
    if len(logps) == 1:
        return logps[0]
    return logps