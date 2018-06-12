from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow.python.ops import *


def is_seqence(x):
    if isinstance(x, (tuple, list)):
        return True
    return False


def have_same_length(x, y):
    if not is_seqence(x):
        raise TypeError("x is not sequence")
    if not is_seqence(y):
        raise TypeError("y is not sequence")

    if not len(x) == len(y):
        return False
    return True


def get_decoder_initial_state(encoder_states,
                              num_layers,
                              encoder_type="bi-directional",
                              decoder_zero_state_fn=None,
                              method="direct-pass",
                              scope=None,
                              num_units,
                              dtype=tf.float32,
                              initializer=None):
    
    raise NotImplementedError("not-used")
    
    if not is_seqence(encoder_states) or \
            len(encoder_states) != num_layers:
        raise ValueError("encoder_states must be sequences "
                         "of encoder states, where len(states) "
                         "equals number of layers, found "
                         "encoder_states to be %s, and "
                         "len(states) %d != %d" % (type(encoder_states),
                                                   len(encoder_states),
                                                   num_layers))

    if encoder_type not in ["bi-directional", "uni-directional"]:
        raise ValueError("%s not recognized" % encoder_type)
    if method not in ["direct-pass", "linear-projection"]:
        raise ValueError("%s not recognized" % method)


    decoder_initial_states = []
    if method == "linear-projection":
        # linear-projection:
        
        # requirements:
        #   1. bidirectional encoders
        #   2. num_encoder_layers = num_decoder_layers

        # linearly project forward and backward cell states
        # into one single state, applied per layer

        # each encoder_state should be list of [fw_state, bw_state]
        if not is_seqence(encoder_states[0]) and len(encoder_states[0]) == 2:
            raise ValueError("linear-projection is for bidirectional encoder")

        for layer_id in range(num_layers):
            with tf.variable_scope(
                    scope or "encoder_scope", reuse=layer_id != 0):
                # Define weights and biases to reduce
                # the cell and reduce the state
                w_reduce_c = tf.get_variable(
                    'w_reduce_c', [num_units * 2, num_units],
                    dtype=dtype, initializer=initializer)
                w_reduce_h = tf.get_variable(
                    'w_reduce_h', [num_units * 2, num_units],
                    dtype=dtype, initializer=initializer)
                bias_reduce_c = tf.get_variable(
                    'bias_reduce_c', [num_units],
                    dtype=dtype, initializer=initializer)
                bias_reduce_h = tf.get_variable(
                    'bias_reduce_h', [num_units],
                    dtype=dtype, initializer=initializer)

                # Apply linear layer
                # Concatenation of fw and bw cell
                cell_states = [encoder_states[0][layer_id],
                               encoder_states[1][layer_id]]
                old_c = tf.concat_v2(axis=1,
                                     values=[st.c for st in cell_states])
                # Concatenation of fw and bw state
                old_h = tf.concat_v2(axis=1,
                                     values=[st.h for st in cell_states])
                # Get new cell from old cell
                new_c = tf.nn.relu(
                    tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
                # Get new state from old state
                new_h = tf.nn.relu(
                    tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
                
                # Return new cell and state
                decoder_initial_states.append(
                    tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h))

