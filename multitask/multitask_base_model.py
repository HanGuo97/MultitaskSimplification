from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from utils import misc_utils
from multitask import multitask_utils
from multitask import sharing_dicts_utils
from pointer_model.batcher import Batcher


MIXING_RATIOS_BASE = 10
tf.logging.set_verbosity(tf.logging.INFO)


class MultitaskBaseModel(object):
    """Multitask Model"""
    def __init__(self,
                 names,
                 all_hparams,
                 mixing_ratios,
                 model_creators,
                 logdir,
                 soft_sharing_coef=None,
                 data_generators=None,
                 val_data_generator=None,
                 *args, **kargs):
        # check lengths match
        if not len(names) == len(all_hparams):
            raise ValueError("names and all_hparams size mismatch")
        if not len(names) == len(model_creators):
            raise ValueError("names and model_creators size mismatch")
        if data_generators and not (
                len(names) == len(data_generators) and
                isinstance(data_generators, MultitaskBatcher)):
            raise ValueError("names and data_generators shape mismatch or "
                             "data_generators is not MultitaskBatcher")

        # check mixing ratios and MTL
        if len(names) == 1 and mixing_ratios is not None:
            raise ValueError("if running single model, set mixing_ratios None")
        if mixing_ratios is not None:
            if len(names) != len(mixing_ratios) + 1:
                raise ValueError("names and mixing_ratios + 1 size mismatch")
            checked_mixing_ratio = [
                _assert_mixing_ratio_compatability(mr) for mr in mixing_ratios]
            print("With Base %d Scaled mixing batch ratios are " %
                MIXING_RATIOS_BASE, checked_mixing_ratio)

        if not soft_sharing_coef or soft_sharing_coef < 1e-6:
            raise ValueError("soft_sharing_coef too small")

        # misc check
        if not all([callable(mc) for mc in model_creators]):
            raise TypeError("Expect model_creator to be callable")
        misc_utils.assert_all_same(all_hparams, attr="batch_size")

        

        if len(names) == 1:
            sharing_dicts = [
                sharing_dicts_utils.sharing_dict_soft]
            # won't be used anyway
            soft_sharing_params = [
                sharing_dicts_utils.Layered_Shared_Params]
        else:
            sharing_dicts = [
                sharing_dicts_utils.sharing_dict_soft,
                sharing_dicts_utils.sharing_dict_soft]
            soft_sharing_params = [
                sharing_dicts_utils.Layered_Shared_Params,
                sharing_dicts_utils.E1D2_Shared_Params]

        # make sure sharing dictionaries are the same
        misc_utils.assert_all_same(sharing_dicts)
        if len(names) != 1:
            # sharing dicts and soft-sharing params for main models
            # in decode models or baseine, only one sharing_dict
            sharing_dicts = [sharing_dicts[0]] + sharing_dicts
            # main model's soft-sharing params should be
            # the union of two soft-sharing params
            # and only one in decode models
            soft_sharing_params = [misc_utils.union_lists(
                soft_sharing_params)] + soft_sharing_params

        # create MTL scopes
        MTL_scope = multitask_utils.MTLScope(names, sharing_dicts)

        # build models
        graph = tf.Graph()
        with graph.as_default():
            # global step shared across all models
            global_step = tf.Variable(
                0, name='global_step', trainable=False)
            models, steps = self._build_models(
                names=names,
                MTL_scope=MTL_scope,
                all_hparams=all_hparams,
                global_step=global_step,
                model_creators=model_creators,
                soft_sharing_coef=soft_sharing_coef,
                soft_sharing_params=soft_sharing_params,
                *args, **kargs)

            saver = tf.train.Saver(max_to_keep=20)

            save_path = None
            summary_dir = None
            summary_writer = None
            if logdir is not None:
                # e.g. model-113000.meta
                save_path = os.path.join(logdir, "model")
                summary_dir = os.path.join(logdir, "summaries")
                summary_writer = tf.summary.FileWriter(summary_dir)

            if not len(names) == len(models):
                raise ValueError("built `models` have mismatch shape, names")
        

        self._sess = None
        self._graph = graph
        self._steps = steps
        self._names = names
        self._models = models
        self._MTL_scope = MTL_scope
        self._all_hparams = all_hparams
        self._global_step = global_step
        self._data_generators = data_generators
        self._val_data_generator = val_data_generator
        
        self._mixing_ratios = mixing_ratios
        self._sharing_dicts = sharing_dicts
        self._soft_sharing_coef = soft_sharing_coef
        self._soft_sharing_params = soft_sharing_params

        self._saver = saver
        self._logdir = logdir
        self._save_path = save_path
        self._summary_dir = summary_dir
        self._summary_writer = summary_writer
        

    def _build_models(self,
                      names,
                      MTL_scope,
                      all_hparams,
                      global_step,
                      model_creators,
                      soft_sharing_coef,
                      soft_sharing_params,
                      vocab,
                      # kept for compatability
                      *args, **kargs):
        models = []
        steps = {"GlobalStep": 0}
        for name, hparams, model_creator, soft_sharing_param in \
                zip(names, all_hparams, model_creators, soft_sharing_params):
            
            print("Creating %s \t%s Model" % (model_creator.__name__, name))
            # this returns a object with scopes as attributes
            scope = MTL_scope.get_scopes_object(name)
            model = model_creator(
                hparams, vocab,
                global_step=global_step,
                name=name, scope=scope,
                soft_sharing_coef=soft_sharing_coef,
                soft_sharing_params=soft_sharing_param)

            with tf.variable_scope(name):
                model.build_graph()
            models.append(model)
            steps[name] = 0

            # reuse variables
            # actually, not necessary
            MTL_scope.reuse_all_shared_variables()

        return models, steps

            

    def initialize_or_restore_session(self, ckpt_file=None):
        """Initialize or restore session

        Args:
            ckpt_file: directory to specific checkpoints
        """
        # restore from lastest_checkpoint or specific file
        with self._graph.as_default():
            self._sess = tf.Session(
                graph=self._graph, config=misc_utils.get_config())
            self._sess.run(tf.global_variables_initializer())

            if self._logdir or ckpt_file:
                # restore from lastest_checkpoint or specific file if provided
                misc_utils.load_ckpt(saver=self._saver,
                                     sess=self._sess,
                                     ckpt_dir=self._logdir,
                                     ckpt_file=ckpt_file)
                return

    
    def _run_train_step(self, batch, model_idx):
        # when running non-major task
        # the regularized model is the major task
        if model_idx != 0:
            reg_model_idx = 0
            
            # when running the auxiliary model
            # the soft-shared parameters should be those
            # that used between this pair model main-aux parameters
            # e.g. for SNLI vs. CNNDM , use SNLI's soft-params
            filtering_fn = lambda name: (
                name in self._soft_sharing_params[model_idx])

        else:  # when model_idx == 0
            # for 3-way models
            # when running second or third model
            # reg_model is the first model
            # when running the first model
            # the reg_model is either second or third model
            reg_model_idx = 2 if self._steps[self._names[0]] % 2 else 1
                
            # when running the main model
            # the soft-shared parameters should be those
            # that used between this pair model main-aux parameters
            # e.g. for CNNDN vs. SNLI, use SNLI's soft-params
            filtering_fn = lambda name: (
                name in self._soft_sharing_params[reg_model_idx])
        
        
        return self._models[model_idx].run_train_step(
            sess=self._sess, batch=batch,
            reg_model_name=self._names[reg_model_idx],
            reg_filtering_fn=filtering_fn,
            all_scopes=self._MTL_scope.all_scopes)

    def run_train_step(self):
        model_idx = self._task_selector(self.global_step)
        model_name = self._names[model_idx]

        # get data batch
        data_batch = self._data_generators.next_batch(model_idx)
        # run one step
        train_step_info = self._run_train_step(data_batch, model_idx)
        # increment train step
        self._steps[model_name] += 1
        self._steps["GlobalStep"] += 1
        # print info and write summary
        # and return the loss for debug usages
        return self._log_train_step_info(train_step_info)

    def _log_train_step_info(self, train_step_info):
        # log statistics
        loss = train_step_info["loss"]
        summaries = train_step_info["summaries"]
        train_step = self._steps["GlobalStep"]
        self._summary_writer.add_summary(summaries, train_step)

        if train_step % 100 == 0:
            self._summary_writer.flush()

        if not np.isfinite(loss):
            self.save_session()
            raise Exception("Loss is not finite. Stopping.")

        # print statistics
        step_msg = "loss: %f step %d " % (loss, train_step)

        for key, val in self._steps.items():
            step_msg += "%s %d " % (key, val)

        tf.logging.info(step_msg)
        return loss

    def run_eval_step(self, model_idx=0):
        # usually only use the main model
        model_name = self._names[model_idx]
        # get data batch
        val_data_batch = self._val_data_generator.next_batch(model_idx)
        # run one step
        val_step_info = self._models[model_idx].run_eval_step(
            sess=self._sess, batch=val_data_batch)
        # NLL loss not included
        val_nll_loss = val_step_info["nll_loss"]

        return val_nll_loss

    def _task_selector(self, step):
        if self._mixing_ratios is None:
            return 0

        return _task_selector_for_three(
            self._mixing_ratios[0], self._mixing_ratios[1], step)

    def save_session(self):
        self._saver.save(self._sess,
            save_path=self._save_path,
            global_step=self.global_step)

    @property
    def global_step(self):
        return self._steps["GlobalStep"]

    @property
    def sess(self):
        return self._sess

    @property
    def graph(self):
        return self._graph

    @property
    def logdir(self):
        return self._logdir

    def run_encoder(self, sess, batch, model_idx=0):
        return self._models[model_idx].run_encoder(sess, batch)

    def decode_onestep(self,
                       sess,
                       batch,
                       latest_tokens,
                       enc_states,
                       dec_init_states,
                       prev_coverage,
                       model_idx=0):
        return self._models[model_idx].decode_onestep(sess, batch,
            latest_tokens, enc_states, dec_init_states, prev_coverage)


class MultitaskBatcher(object):
    """Decorator for Batcher for multiple models"""
    def __init__(self, data_paths, vocabs, hps, single_pass):
        if not len(data_paths) == len(vocabs):
            raise ValueError("data_paths and vocabs size mismatch")
        
        batchers = []
        for data_path, vocab in zip(data_paths, vocabs):
            batcher = Batcher(data_path, vocab, hps, single_pass)
            batchers.append(batcher)

        self._vocabs = vocabs
        self._batchers = batchers
        self._data_paths = data_paths

    def next_batch(self, batcher_idx=0):
        return self._batchers[batcher_idx].next_batch()

    def __len__(self):
        return len(self._batchers)


# ================================================
# some utility functions
# ================================================
def _task_selector_for_three(mixing_ratio_1, mixing_ratio_2, step):
    if mixing_ratio_1 <= 0.01:
        raise ValueError("mr_1 too small")

    if mixing_ratio_2 <= 0.01:
        raise ValueError("mr_2 too small")

    left_over = step % MIXING_RATIOS_BASE
    task_one_boundary = (MIXING_RATIOS_BASE -
        int(MIXING_RATIOS_BASE * (mixing_ratio_1 + mixing_ratio_2)))

    task_two_boundary = (MIXING_RATIOS_BASE -
        int(MIXING_RATIOS_BASE * mixing_ratio_2))

    if 0 <= left_over < task_one_boundary:
        return 0
    elif task_one_boundary <= left_over < task_two_boundary:
        return 1
    else:
        return 2


def _assert_mixing_ratio_compatability(mixing_ratio):
    if not isinstance(mixing_ratio, (int, float)):
        raise TypeError("%s should be int or float, found "
            % mixing_ratio, type(mixing_ratio))
    result = mixing_ratio * MIXING_RATIOS_BASE
    if not int(result) == result:
        raise ValueError("%s x %s = %s are not integers" %
            (mixing_ratio, MIXING_RATIOS_BASE, result))

    if result <= 0.01:
        raise ValueError("MixingRatio too small")

    return int(result)
