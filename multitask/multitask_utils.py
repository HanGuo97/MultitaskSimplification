from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
SHARED_SCOPE_PREFIX = "Shared"
SCOPE_LIST = ['WordEmb',
              'EncoderFW',
              'Projection',
              'Attention',
              'EncoderBW',
              'Decoder',
              'Pointer']



class MTLOutOfRangeError(tf.errors.OutOfRangeError):
    """Wraps tf.errors.OutOfRangeError with model_idx"""
    def __init__(self, node_def, op, message, model_idx):
        super(MTLOutOfRangeError, self).__init__(
            node_def=node_def, op=op, message=message)
        self.model_idx = model_idx


class MTLScope(object):
    def __init__(self,
                 model_names=None,
                 sharing_dicts=None):
        if not isinstance(model_names, (list, tuple)):
            raise TypeError("model_names should be list of dicts")
        if not isinstance(sharing_dicts, (list, tuple)):
            raise TypeError("sharing_dicts should be list of dicts")
        if not len(model_names) == len(sharing_dicts):
            raise ValueError("model_names and sharing_dicts shape mismatch")

        # make sure variables that are shared used the same scope instance
        # by tracking previously created scopes

        # all_scopes keep track of all variable scopes, and store scopes
        # in the format e.g. {
        #   ModelA_Enc_layer1: variable_scope,
        #   ModelA_Enc_layer1: variable_scope
        #   ModelB_Enc_layer2: ... }
        all_scopes = {}

        # model_scopes keep track of variables scopes for each model
        # in the format e.g. {
        #   ModelA:{Encoder: [variable_scope, variable_scope ...] },
        #   ModelB: {...}}
        model_scopes = {}

        for var_name in SCOPE_LIST:
            for model_name, sharing_dict in zip(model_names, sharing_dicts):
                # set ModelName
                model_scopes.setdefault(model_name,
                    {"Model": create_scope(model_name)})

                # raise exception if the sharing dict is in wrong format
                if var_name not in sharing_dict.keys():
                    raise ValueError(
                        "model %s variable %s not in sharing_dict" %
                        (model_name, var_name))
                
                # ensure will_share is a tuple of True or False
                if not is_sequence(sharing_dict[var_name]):
                    this_is_sequence = False
                    will_share_these_var = [sharing_dict[var_name]]
                else:
                    this_is_sequence = True
                    will_share_these_var = sharing_dict[var_name]

                for idx, will_shar_this_var in enumerate(will_share_these_var):
                    if will_shar_this_var:
                        # e.g. "Shared_Encoder_1" and "Shared_Encoder"
                        (scope_name_without_idx,
                         scope_name_maybe_with_idx) = (
                            get_scope_names_tuple(SHARED_SCOPE_PREFIX,
                                var_name, idx, this_is_sequence))

                        # when sharing variables, get previou created ones
                        # if scopes with the same same was created, otherwise
                        # create a new one
                        if scope_name_maybe_with_idx in all_scopes.keys():
                            # reuse previous shared scope
                            scope = all_scopes[scope_name_maybe_with_idx]
                        else:  # create a new one
                            scope = create_scope(scope_name_maybe_with_idx)
                            all_scopes[scope_name_maybe_with_idx] = scope
            
                    else:
                        # e.g. "ModelA_Encoder_1" and "ModelA_Encoder"
                        (scope_name_without_idx,
                         scope_name_maybe_with_idx) = (
                            get_scope_names_tuple(
                                model_name, var_name, idx, this_is_sequence))
                        
                        # not sharing variables
                        scope = create_scope(scope_name_maybe_with_idx)
                        all_scopes[scope_name_maybe_with_idx] = scope

                    # model_scopes are indexed by:
                    #   1. model_name (dictionary)
                    #       2. var_name (dictionary)
                    #           3. index (list)
                    if this_is_sequence:
                        model_scopes[model_name].setdefault(var_name, [])
                        model_scopes[model_name][var_name].append(scope)
                        if model_scopes[model_name][var_name].index(scope) != idx:
                            raise ValueError("Index messed up ",
                                model_scopes[model_name][var_name])
                    else:
                        model_scopes[model_name][var_name] = scope
                    
        
        self.all_scopes = all_scopes
        self.model_scopes = model_scopes
    
    def print_scope_names(self):
        for model_name, scopes in self.model_scopes.items():
            msg = "%s" % model_name
            for var, scope in scopes.items():
                if is_sequence(scope):
                    info = ["\t%s %s" % (s.name, s.reuse) for s in scope]
                    info = "".join(info)
                else:
                    info = "\t%s %s" % (scope.name, scope.reuse)

                msg += "\n%s:\t %s" % (var, info)
            print(msg + "\n")
    
    def reuse_all_shared_variables(self):
        for scope in self.all_scopes.values():
            if SHARED_SCOPE_PREFIX in scope.name:
                scope.reuse_variables()

    def get_scopes_list(self):
        return list(self.model_scopes.values())

    def get_scopes(self):
        raise Exception("Please switch to get_scopes_list")
    
    def get_scopes_object(self, model_name):
        if model_name not in self.model_scopes.keys():
            raise ValueError(
                "model_name %s not in model_names" % model_name)
            
        return type("", (object,), self.model_scopes[model_name])()


def create_scope(name):
    with tf.variable_scope(name) as scope:
        pass
    return scope


def is_sequence(X):
    if isinstance(X, (list, tuple)):
        return True
    return False


def get_scope_names_tuple(name_prefix, var_name, idx, this_is_sequence):
    scope_name_without_idx = "_".join([name_prefix, var_name])
    scope_name_with_idx = "_".join([name_prefix, var_name, str(idx)])

    if not this_is_sequence:
        return [scope_name_without_idx, scope_name_without_idx]
    else:
        return [scope_name_without_idx, scope_name_with_idx]