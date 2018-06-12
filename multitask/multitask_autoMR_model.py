from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from multitask import bandits
from multitask.multitask_base_model import MultitaskBaseModel


class MultitaskAutoMRModel(MultitaskBaseModel):
    """
    Multitask model with automatic task selection

    Build a TaskSelector object that keeps track of
    previous val-loss when running on task S, and
    choose which task to run by sampling from:

        S_t ~ P(S_t | history of validation loss)

    where P is modeled as a boltzmann distribution

        P(S | history) = softmax(history)

    and S is kept constant until new validation loss
    is available, that is, every 10 or so steps
    
    Q Score should be negative loss thus lower is better
    high initial Q for being "optimistic under uncertainty"

    """
    def _build_models(self,
                      names,
                      selector_Q_initial,
                      alpha=0.3,
                      temperature_anneal_rate=None,
                      *args, **kargs):
        
        self._task_selector_actions = names
        self._TaskSelector = bandits.MultiArmedBanditSelector(
            num_actions=len(names),
            initial_weight=selector_Q_initial,
            update_rate_fn=lambda step: alpha,  # constant update
            reward_shaping_fn=lambda reward, hist: reward,  # no shaping
            temperature_anneal_rate=temperature_anneal_rate)
        print("Initial TaskSelector Q_score: %.1f, "
              "and temperature anneal rate %.5f"
              % (selector_Q_initial,
              self._TaskSelector._temperature_anneal_rate or 1.0))

        # initial task will be main task
        self._current_task_index = 0

        # normal building models
        return super(MultitaskAutoMRModel, self)._build_models(
            names=names, *args, **kargs)

    def update_TaskSelector_Q_values(self, Q_score):
        
        self._TaskSelector.update(
            reward=Q_score,
            chosen_arm=self._current_task_index)

        # sample a new task to run
        self._current_task_index, _ = (
            self._TaskSelector.sample(step=self.global_step))

        # print info
        print("\n\n\n")
        print("New Q_score: %.3f" % Q_score)
        print("ChosenTask: %d" % self._current_task_index)
        for idx, val in enumerate(self._TaskSelector.arm_weights):
            print("%s/Expected_Q_Value: %.3f"
                % (self._task_selector_actions[idx], val))
        print("\n\n\n")

    def _task_selector(self, step):
        # override parent method
        # step argument is kept for compatability
        return self._current_task_index

    def save_selector(self):
        # additionally save the selector
        selector_dir = os.path.join(self._logdir, "mab_selector.pkl")
        self._TaskSelector.save(selector_dir)

    def load_selector(self):
        # additionally restore the selector
        selector_dir = os.path.join(self._logdir, "mab_selector.pkl")
        # if not exist, skip this
        if os.path.exists(selector_dir):
            self._TaskSelector.load(selector_dir)
    
    def save_session(self):
        self.save_selector()
        return super(MultitaskAutoMRModel, self).save_session()


    def initialize_or_restore_session(self, *args, **kargs):
        self.load_selector()
        return super(MultitaskAutoMRModel, self).initialize_or_restore_session(*args, **kargs)
