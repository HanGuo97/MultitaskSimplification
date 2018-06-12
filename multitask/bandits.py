from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import warnings
import numpy as np
from namedlist import namedlist

Q_Entry = namedlist("Q_Entry", ("Value", "Count"))


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def gradient_bandit(old_Q, reward, alpha):
    new_Q = old_Q + alpha * (reward - old_Q)
    return new_Q


def convert_to_one_hot(action_id, action_space):
    return np.eye(action_space)[action_id]


def boltzmann_exploration(Q_values, temperature=1.0):
    # for numerical stability, add 1e-7
    Q_probs = softmax(Q_values, theta=1 / (temperature + 1e-7))
    action = np.random.choice(len(Q_probs), p=Q_probs)
    return action, Q_probs


class MultiArmedBanditSelector(object):
    def __init__(self,
                 num_actions,
                 initial_weight,
                 update_rate_fn,
                 reward_shaping_fn,
                 initial_temperature=1.0,
                 temperature_anneal_rate=None):
        """
        Args:
            update_rate_fn: fn(step) --> Real
                a function that takes `step` as input, and produce
                real value, the gradent bandit update rate.
                Common functions include:
                    1. (constant update) lambda step: CONSTANT
                    2. (average of entire history): lambda step: 1 / (step + 1)

            reward_shaping_fn: fn(reward, histories) --> Real
                a function that takes current and histories of rewards
                as inputs and produce real value, the reward to be fed into
                the bandits algorithm
                Common functions include:
                    1. lambda reward, hist: reward / CONSTANT
                    2. lambda reward, hist: [reward - mean(hist)] / std(hist)
        """
        if not callable(update_rate_fn):
            raise TypeError("`update_rate_fn` must be callable")
        if not callable(reward_shaping_fn):
            raise TypeError("`reward_shaping_fn` must be callable")

        self._Q_entries = [
            # intial Count = 1 because of `initial_weight`
            Q_Entry(Value=initial_weight, Count=1)
            for _ in range(num_actions)]
        self._num_actions = num_actions
        self._update_rate_fn = update_rate_fn
        self._reward_shaping_fn = reward_shaping_fn

        self._temperature = initial_temperature
        self._temperature_anneal_rate = temperature_anneal_rate
        
        self._sample_histories = []
        self._update_histories = []


    def sample(self, step=0):
        temperature_coef = (  # tau x rate^step
            np.power(self._temperature_anneal_rate, step)
            if self._temperature_anneal_rate is not None else 1.)

        chosen_action, Q_probs = boltzmann_exploration(
            Q_values=np.asarray(self.arm_weights),
            temperature=self._temperature * temperature_coef)

        self._sample_histories.append([Q_probs, chosen_action])

        return chosen_action, Q_probs

    def update(self, reward, chosen_arm):
        # uses sampling, set weights = 1
        if not isinstance(chosen_arm, int):
            raise ValueError("chosen_arm must be integers")
        if not chosen_arm < self._num_actions:
            raise ValueError("chosen_arm out of range")

        step_size = self._update_rate_fn(
            self._Q_entries[chosen_arm].Count)
        shaped_reward = self._reward_shaping_fn(
            reward, self.reward_histories)

        new_Q = gradient_bandit(reward=shaped_reward, alpha=step_size,
                                old_Q=self._Q_entries[chosen_arm].Value)
        
        self._Q_entries[chosen_arm].Count += 1
        self._Q_entries[chosen_arm].Value = new_Q
        self._update_histories.append([reward, chosen_arm, shaped_reward])


    @property
    def arm_weights(self):
        return [Q.Value for Q in self._Q_entries]

    @property
    def step_counts(self):
        return np.sum([Q.Count for Q in self._Q_entries])

    @property
    def reward_histories(self):
        # at the start, the update_histories is empty
        # to avoid nan, we will force set this to 0
        if len(self._update_histories) == 0:
            return [0.0]

        return [hist[0] for hist in self._update_histories]

    def save(self, file_dir):
        with open(file_dir + "._Q_entries", "wb") as f:
            pickle.dump(self._Q_entries, f, pickle.HIGHEST_PROTOCOL)

        with open(file_dir + "._sample_histories", "wb") as f:
            pickle.dump(self._sample_histories, f, pickle.HIGHEST_PROTOCOL)

        with open(file_dir + "._update_histories", "wb") as f:
            pickle.dump(self._update_histories, f, pickle.HIGHEST_PROTOCOL)

        print("INFO: Successfully Saved MABSelector to ", file_dir)

    def load(self, file_dir):
        warnings.warn("num_actions are *NOT* checked")
        for suffix in ["._Q_entries",
                       "._sample_histories",
                       "._update_histories"]:
            if not os.path.exists(file_dir + suffix):
                raise ValueError("%s File not exist ", suffix)

        with open(file_dir + "._Q_entries", "rb") as f:
            Q_values = pickle.load(f)

        with open(file_dir + "._sample_histories", "rb") as f:
            sample_histories = pickle.load(f)

        with open(file_dir + "._update_histories", "rb") as f:
            update_histories = pickle.load(f)
        
        self._Q_entries = Q_values
        self._sample_histories = sample_histories
        self._update_histories = update_histories

        print("INFO: Successfully Loaded %s from %s" %
            (self.__class__.__name__, file_dir))
