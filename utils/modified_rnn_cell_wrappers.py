from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell

# Hashable List
NUM_LAYERS = 2
MultiRNNLSTMStateTuple = namedtuple("MultiRNNLSTMStateTuple",
                                    ("Layer0", "Layer1"))


def to_MultiRNNLSTMStateTuple(states):
    if isinstance(states, MultiRNNLSTMStateTuple):
        return states

    if not isinstance(states, (list, tuple)):
        raise TypeError(
            "Expected states to be list, found ", type(states))

    if not len(states) == NUM_LAYERS:
        raise ValueError(
            "Only %d layers are supported now, found %d"
            % (NUM_LAYERS, len(states)))

    return MultiRNNLSTMStateTuple(* states)


class SingleRNNCell(rnn_cell.RNNCell):
    """Cell Wrapper with Scope"""
    def __init__(self, cell, cell_scope):
        # use if in TF1.0 or afterwards
        super(SingleRNNCell, self).__init__()
        self._cell = cell
        self._cell_scope = cell_scope

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        # customized scope
        with vs.variable_scope(self._cell_scope or "Cell"):
            outputs, new_state = self._cell(inputs, state)

        return outputs, new_state


class MultiRNNCell(rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=True, cell_scopes=None):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        # use if in TF1.0 or afterwards
        super(MultiRNNCell, self).__init__()
        if not cells:
            raise ValueError(
                "Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError(
                "cells must be a list or tuple, but saw: %s." % cells)
        if (not isinstance(cell_scopes, (tuple, list)) and
                len(cell_scopes) == len(cells)):
            raise ValueError(
                "scopes should be a list with same shape as cells")
        if not len(cells) == NUM_LAYERS:
            raise ValueError("Only two layer Cells are supported")

        self._cells = cells
        self._state_is_tuple = state_is_tuple
        self._cell_scopes = cell_scopes
        if not state_is_tuple:
            if any(nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError(
                    "Some cells return tuples of states, but the flag "
                    "state_is_tuple is not set.  State sizes are: %s"
                    % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            state_size = tuple(cell.state_size for cell in self._cells)
            return MultiRNNLSTMStateTuple(* state_size)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        # overwrite 0.12 styles with 1.4 style
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                zero_state = tuple(cell.zero_state(batch_size, dtype)
                                   for cell in self._cells)
                # wrap list of states with hashable tuple
                return MultiRNNLSTMStateTuple(* zero_state)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else
                # fancy
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            # customized scope
            with vs.variable_scope(self._cell_scopes[i] or "Cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, "
                            "but received: %s" % (len(self.state_size), state))
                    if not isinstance(state, MultiRNNLSTMStateTuple):
                        raise TypeError(
                            "Expected state to be MultiRNNLSTMStateTuple, "
                            "found ", type(state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

        if self._state_is_tuple:
            # wrap list of states with hashable tuple
            new_states = tuple(new_states)
            new_states = MultiRNNLSTMStateTuple(* new_states)
        else:
            # use concat_v2 for consistency with TF1.0
            new_states = array_ops.concat(new_states, 1)

        return cur_inp, new_states
