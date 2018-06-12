from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def _varname_to_plname(varname):
        # e.g. Shared_AttentionScope/Wa:0
        # should become Shared_AttentionScope_Wa_pl
        plname = varname.replace("/", "_")
        plname = plname.replace(":0", "")
        plname = "_".join([plname, "pl"])
        return plname


def get_regularzation_loss(filtering_fn, scope, coef=0.0001):
    """calculate regularization loss as soft-sharing constraint"""
    if not callable(filtering_fn):
        raise TypeError(
            "Expected `filtering_fn` to be callable, found ",
            type(filtering_fn))

    with tf.variable_scope(scope.Model.name):
        shared_vars = [
            v for v in tf.trainable_variables() if filtering_fn(v.name)]

        shared_var_pls = [
            tf.placeholder(tf.float32,
                           shape=shared_var.get_shape(),
                           name=_varname_to_plname(shared_var.name))
            for shared_var in shared_vars]
        
        
        if not len(shared_vars) == len(shared_var_pls):
            raise ValueError(
                "shared_vars and shared_var_pls have different lengths")
        
        # total regularization loss
        reg_loss = 0
        # mapping from placeholder name to var name
        reg_pl_names_dict = {}
        for var_pl, var in zip(shared_var_pls, shared_vars):
            reg_loss += tf.nn.l2_loss(var_pl - var)
            reg_pl_names_dict[var_pl.name] = var.name
        
        reg_loss = tf.multiply(coef, reg_loss)

    return reg_loss, reg_pl_names_dict


def calc_regularization_loss(filtering_fn,
                             reg_pl_names_dict,
                             reg_model_name,
                             feed_dict,
                             sess,
                             all_scopes=None):
    """Calculate regularization loss

    Args:
        filtering_fn:
            callable(reg_param_name) --> boolean
            whether to add regularization loss on this param
            if False, then reg_placeholder will be filled
            with same param effectively making regulaization loss 0
        reg_pl_names_dict:
            dictionary mapping placeholder_name to param_name
        reg_model_name:
            name of the model to be regularized w.r.t
        feed_dict:
            feed_dict to be used in sess.run
        all_scopes:
            all parameter scopes, used to check whether the
            new parameter name is valid

    """
    if not callable(filtering_fn):
        raise TypeError("`filtering_fn` should be callable, found ",
            type(filtering_fn).__name__)

    for reg_pl_name, reg_param_name in reg_pl_names_dict.items():
        # decide whether a parameter is to be softly shared
        # for those not softly shared at this timestep,
        # but still have a placeholder for parameters to be shared
        # we just let the parameters to be regularized bt itself,
        # or setting Loss = || param_i - param_i ||
        # which effectively means Loss = 0
        # there are obviously better ways to approach this
        # e.g. using conditional graph, but this is a bit tricky
        # to implement and not much speed gain anyway
        if filtering_fn(reg_param_name):
            changed_reg_param_name = "_".join(
                [reg_model_name] + reg_param_name.split("_")[1:])

        else:
            # this will make reg_loss == 0
            changed_reg_param_name = reg_param_name

        # just to make sure the new name is within scopes
        if (all_scopes and changed_reg_param_name.split("/")[0]
                not in all_scopes):
            raise ValueError("%s not in all scopes"
                % changed_reg_param_name.split("/")[0])

        # add regularization terms into feed_dict
        feed_dict[reg_pl_name] = sess.run(changed_reg_param_name)

    return feed_dict
