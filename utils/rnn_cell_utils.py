"""https://github.com/tensorflow/nmt"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


def _single_cell(unit_type,
                 num_units,
                 mode="train",
                 dropout=None,
                 residual_connection=False,
                 *args, **kargs):
    """Create an instance of a single RNN cell."""

    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units, *args, **kargs)

    elif unit_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell(
            num_units, *args, **kargs)

    elif unit_type == "layer_norm_lstm":
        single_cell = tf.nn.rnn_cell.LayerNormBasicLSTMCell(
            num_units, layer_norm=True, *args, **kargs)

    elif unit_type == "classical_lstm":
        single_cell = tf.nn.rnn_cell.LSTMCell(
            num_units, *args, **kargs)

    else:
        raise ValueError("Unknown unit type %s !" % unit_type)

    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    if dropout is not None:
        dropout = dropout if mode == "train" else 0.0
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))

        print("Using Dropout of dropout_keep rate %.2f" % (1.0 - dropout))

    # Residual
    if residual_connection:
        single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell)


    return single_cell


def _cell_list(unit_type,
               num_units,
               num_layers,
               mode="train",
               dropout=None,
               num_residual_layers=0,
               single_cell_fn=None,
               *args, **kargs):
    """Create a list of RNN cells."""
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            mode=mode,
            dropout=dropout,
            residual_connection=(i >= num_layers - num_residual_layers),
            *args, **kargs)
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    mode,
                    dropout=None,
                    num_residual_layers=0,
                    single_cell_fn=None,
                    cell_wrapper=None,
                    cell_scope=None,
                    *args, **kargs):
    """Create multi-layer RNN cell.

    Args:
      unit_type: string representing the unit type, i.e. "lstm".
      num_units: the depth of each unit.
      num_layers: number of cells.
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
        cells in the returned list will be wrapped with `ResidualWrapper`.
      dropout: floating point value between 0.0 and 1.0:
        the probability of dropout.  this is ignored if `mode != TRAIN`.
      mode: either tf.contrib.learn.TRAIN/EVAL/INFER
      single_cell_fn: allow for adding customized cell.
        When not specified, we default to model_helper._single_cell
    Returns:
      An `RNNCell` instance.
    """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           mode=mode,
                           dropout=dropout,
                           num_residual_layers=num_residual_layers,
                           single_cell_fn=single_cell_fn,
                           *args, **kargs)

    if cell_wrapper and not callable(cell_wrapper):
        raise TypeError("Expect `cell_wrapper` to be callable, "
                        "found ", type(cell_wrapper))

    if len(cell_list) == 1:  # Single layer.
        if not cell_wrapper:
            return cell_list[0]
        return cell_wrapper(cell=cell_list[0], cell_scope=cell_scope)
        
    else:  # Multi layers
        if not cell_wrapper:
            return tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell_wrapper(cells=cell_list, cell_scopes=cell_scope)
