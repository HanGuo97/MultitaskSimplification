import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from pointer_model import data
from evaluation_utils import bleu


def cross_entropy_loss(p_model, targets, loss_mask, batch_size):
    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the
    # probabilities of the gold target words
    # will be list length max_dec_steps containing shape
    # (batch_size)

    # this works when dec_seq_len is undetermined
    dec_seq_len = array_ops.shape(p_model)[1]
    batch_nums = math_ops.range(0, limit=batch_size)
    batch_nums = array_ops.expand_dims(batch_nums, 1)
    batch_nums = array_ops.tile(batch_nums, [1, dec_seq_len])
    
    # time indices
    time_nums = math_ops.range(array_ops.shape(p_model)[1])
    time_nums = array_ops.expand_dims(time_nums, 0)
    time_nums = array_ops.tile(time_nums, [batch_size, 1])
    
    indices = array_ops.stack((batch_nums, time_nums, targets), axis=-1)
    gold_probs = array_ops.gather_nd(p_model, indices)
    raw_loss = - math_ops.log(gold_probs + 1e-6)
    
    # words per sentence
    tokens_per_seq = math_ops.reduce_sum(loss_mask, axis=-1)
    # masked loss
    masked_loss = raw_loss * loss_mask
    # average loss per sequence
    sequence_loss = math_ops.reduce_sum(masked_loss, axis=-1) / tokens_per_seq
    # avergae loss per batch
    loss = math_ops.reduce_mean(sequence_loss)
    
    return loss


def negative_log_likelihood(actions_prob,
                            target_actions,
                            episode_masks,
                            action_space,
                            dtype=dtypes.float32,
                            policy_multipliers=None):
    # exactly equal to `cross_entropy_loss`
    # but simpler
    if policy_multipliers is None:
        # broadcasting will do the rest of the jobs
        policy_multipliers = 1

    # calculate - p(x) log q(x)
    actions_log_prob = math_ops.log(actions_prob + 1e-6)
    target_actions_onehot = array_ops.one_hot(
        indices=target_actions,
        depth=action_space, dtype=dtype)
    nll = - target_actions_onehot * actions_log_prob
    
    # masked NLL, or nll * mask * policy_multipliers
    masked_policy_multipliers = array_ops.expand_dims(
        policy_multipliers * episode_masks, axis=2)
    scaled_masked_nll = nll * masked_policy_multipliers
    
    # sequence and batch NLL
    actions_per_episode = math_ops.reduce_sum(episode_masks, axis=-1)
    sequence_nll = math_ops.reduce_sum(scaled_masked_nll, axis=[1, 2])
    sequence_nll = sequence_nll / actions_per_episode
    batch_nll = math_ops.reduce_mean(sequence_nll)
    return batch_nll


def calc_bleu_rewards(sess,
                      feed_dict,
                      vocabulary,
                      batch_OOVs,
                      target_actions_pl,
                      sampled_actions_pl,
                      policy_multipliers_pl):
    fetches = sess.run(
        fetches={"target_actions": target_actions_pl,
                 "sampled_actions": sampled_actions_pl},
        feed_dict=feed_dict)

    rewards = []
    for target, sampled in zip(fetches["target_actions"].tolist(),
                               fetches["sampled_actions"].tolist()):
        target_actions = data.outputids2words(
            target, vocabulary, batch_OOVs)
        sampled_actions = data.outputids2words(
            sampled, vocabulary, batch_OOVs)

        reward, _, _, _, _, _ = bleu.compute_bleu(
            reference_corpus=[[target_actions]],
            translation_corpus=[sampled_actions],
            max_order=4,
            smooth=True)

        rewards.append(100 * reward)
    
    batch_size = fetches["sampled_actions"].shape[0]
    sequence_lengths = fetches["sampled_actions"].shape[1]

    if not len(rewards) == batch_size:
        raise ValueError("rewards lengths %d "
            "!= batch_size %d" % (len(rewards), batch_size))
    
    # add time dimensions
    rewards = np.expand_dims(rewards, axis=1)
    # tile the time dimension
    rewards = np.tile(rewards, reps=sequence_lengths)
    feed_dict[policy_multipliers_pl] = rewards
    print("BatchRewards: ", np.mean(rewards))

    return feed_dict
