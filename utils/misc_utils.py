from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from time import time
from copy import deepcopy
from contextlib import contextmanager

import tensorflow as tf
from utils import modified_rnn_cell_wrappers as cell_wrappers

FLAGS = tf.app.flags.FLAGS


@contextmanager
def calculate_time(tag):
    start_time = time()
    yield
    print("%s: " % tag, time() - start_time)


def union_lists(lists):
    if not isinstance(lists, (list, tuple)):
        raise TypeError("lists Must be a list of list")

    new_list = deepcopy(lists[0])
    for single_list in lists:
        if not isinstance(single_list, (list, tuple)):
            raise TypeError("single_list Must be a list")
        for list_item in single_list:
            if list_item not in new_list:
                new_list.append(list_item)
    return new_list


def assert_all_same(items, attr=None):
    if not isinstance(items, (list, tuple)):
        raise TypeError("items should be list or tuple")

    if attr is not None:
        if not all(getattr(x, attr) == getattr(items[0], attr) for x in items):
            raise ValueError("items of %s not consistent between items" % attr)
    else:
        if not all(x == items[0] for x in items):
            raise ValueError("items not consistent between items")


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir=None, ckpt_file=None):
    if not ckpt_dir and not ckpt_file:
        return

    if not ckpt_file:
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file is None:
            return

    saver.restore(sess, ckpt_file)
    tf.logging.info("Loaded checkpoint %s" % ckpt_file)
    return ckpt_file


def concate_multi_rnn_cell_states(states, concat_fn, expand_fn):
    if not isinstance(states, (list, tuple)):
        raise TypeError(
            "states should be a list of beam_size, but saw ", type(states))
    if not isinstance(states[0], (list, tuple)):
        raise TypeError(
            "each states[beam_id] should be a list "
            "of multi_rnn states, but saw ", type(states[0]))
    if not isinstance(states[0][0], tf.nn.rnn_cell.LSTMStateTuple):
        raise TypeError("each states[beam_id][layer_id] should be "
                        "LSTMStateTuple, but saw ", type(states[0][0]))
    num_beams = len(states)
    num_layers = len(states[0])
    concat_states = []
    append_states = [
        tf.nn.rnn_cell.LSTMStateTuple(c=[], h=[])
        for _ in range(num_layers)]
    
    # append all cell and hidden states to lists
    for beam_id in range(num_beams):
        for layer_id in range(num_layers):
            append_states[layer_id].c.append(expand_fn(states[beam_id][layer_id].c))
            append_states[layer_id].h.append(expand_fn(states[beam_id][layer_id].h))
    
    # concatenate the list
    for layer_id in range(num_layers):
        concat_c = concat_fn(append_states[layer_id].c)
        concat_h = concat_fn(append_states[layer_id].h)
        concat_states.append(tf.nn.rnn_cell.LSTMStateTuple(c=concat_c, h=concat_h))
    

    return cell_wrappers.to_MultiRNNLSTMStateTuple(concat_states)


def split_multi_rnn_cell_states(states):
    if not isinstance(states, (list, tuple)):
        raise TypeError(
            "states should be a list of beam_size, but saw ", type(states))
    if not isinstance(states[0], (list, tuple)):
        raise TypeError("each states[layer_id] should be "
                        "LSTMStateTuple, but saw ", type(states[0]))
    num_layers = len(states)
    num_beams = states[0].c.shape[0]
    new_states = [[0 for _ in range(num_layers)] for _ in range(num_beams)]
    for layer_id in range(num_layers):
        for beam_id in range(num_beams):
            c = states[layer_id].c[beam_id, :]
            h = states[layer_id].h[beam_id, :]
            new_states[beam_id][layer_id] = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
    return new_states
