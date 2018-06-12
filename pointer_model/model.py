# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""https://github.com/abisee/pointer-generator"""
import time
import numpy as np
import tensorflow as tf

from pointer_model import data

from utils import misc_utils
from utils import rnn_cell_utils
from multitask import soft_sharing_utils as soft_utils
from utils import modified_rnn_cell_wrappers as cell_wrappers

from pointer_model import pg_decoder
from pointer_model import policy_gradient_utils as pg_utils


class SummarizationModel(object):
    def __init__(self,
                 hps,
                 vocab,
                 global_step,
                 name=None,
                 scope=None,
                 soft_sharing_coef=None,
                 soft_sharing_params=None,
                 reinforce=False):
        self._hps = hps
        self._vocab = vocab
        self.global_step = global_step
        self._name = name

        self._scope = scope
        self._soft_sharing_coef = soft_sharing_coef
        self._soft_sharing_params = soft_sharing_params

        if reinforce:
            raise NotImplementedError("Not Tested")
        self._reinforce = reinforce

        self._summaries = []


    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(
            tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(
            tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        if self._hps.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(
                tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(
                tf.int32, [], name='max_art_oovs')

        # decoder part
        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(
                tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if self._hps.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        # let cells be wrapped with Scope
        if self._hps.num_encoder_layers == 1:
            cell_wrapper = cell_wrappers.SingleRNNCell
        else:
            cell_wrapper = cell_wrappers.MultiRNNCell

        with tf.variable_scope("encoder_scope"):
            cell_fws = rnn_cell_utils.create_rnn_cell(
                unit_type="classical_lstm",
                num_units=self._hps.hidden_dim,
                num_layers=self._hps.num_encoder_layers,
                mode=self._hps.mode,
                dropout=self._hps.dropout_rate,
                cell_wrapper=cell_wrapper,
                cell_scope=self._scope.EncoderFW,
                # additional args
                state_is_tuple=True,
                initializer=self.rand_unif_init)

            cell_bws = rnn_cell_utils.create_rnn_cell(
                unit_type="classical_lstm",
                num_units=self._hps.hidden_dim,
                num_layers=self._hps.num_encoder_layers,
                mode=self._hps.mode,
                dropout=self._hps.dropout_rate,
                cell_wrapper=cell_wrapper,
                cell_scope=self._scope.EncoderBW,
                # additional args
                state_is_tuple=True,
                initializer=self.rand_unif_init)

            (encoder_outputs, encoder_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fws, cell_bws, encoder_inputs,
                dtype=tf.float32, sequence_length=seq_len, swap_memory=True)

            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        return encoder_outputs, encoder_states

    def _reduce_states(self, encoder_states):
        """Add to the graph a linear layer to reduce
        the encoder's final FW and BW state into a single
        initial state for the decoder. This is needed because
        the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self._hps.hidden_dim
        num_encoder_layers = self._hps.num_encoder_layers
        num_decoder_layers = self._hps.num_decoder_layers

        # if only one-layer bidirectional encoder
        # decoder_cell_layer_0_init_state = fw_cell_state
        # decoder_cell_layer_1_init_state = bw_cell_state
        # this follows Google's NMT implementation
        if num_encoder_layers == 1:
            if not num_decoder_layers == 2:
                raise NotImplementedError("to support num_decoder_layers == 1 "
                                          "decoder cell states should change from "
                                          "customized multi_rnn_cell_states to allow "
                                          "single layer")

            # encoder_states should be [fw_state, bw_state]
            if not len(encoder_states) == 2:
                raise ValueError("encoder_states must be length 2")
            if not isinstance(encoder_states[0], tf.nn.rnn_cell.LSTMStateTuple):
                raise TypeError("encoder_states[0] must be LSTMStateTuple")

            return cell_wrappers.to_MultiRNNLSTMStateTuple(encoder_states)

        # otherwise, do linear-projection
        decoder_initial_states = []
        for layer_id in range(num_encoder_layers):
            with tf.variable_scope("encoder_scope", reuse=layer_id != 0):
                # Define weights and biases to reduce the cell and reduce the state
                w_reduce_c = tf.get_variable('w_reduce_c', [
                                             hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                w_reduce_h = tf.get_variable('w_reduce_h', [
                                             hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                bias_reduce_c = tf.get_variable(
                    'bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                bias_reduce_h = tf.get_variable(
                    'bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

                # Apply linear layer
                # Concatenation of fw and bw cell
                cell_states = [encoder_states[0][layer_id],
                               encoder_states[1][layer_id]]
                old_c = tf.concat(axis=1,
                                     values=[st.c for st in cell_states])
                # Concatenation of fw and bw state
                old_h = tf.concat(axis=1,
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
        
        return cell_wrappers.to_MultiRNNLSTMStateTuple(decoder_initial_states)


    def _add_decoder(self, inputs, embedding):
        """Add attention decoder to the graph. In train or eval mode,
        you call this once to get output on ALL steps. In decode (beam
        search) mode, you call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings).
          A list of tensors shape (batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of scalar tensors; the generation probabilities
          coverage: A tensor, the current coverage vector
        """
        hps = self._hps
        cells = rnn_cell_utils.create_rnn_cell(
            unit_type="classical_lstm",
            num_units=self._hps.hidden_dim,
            num_layers=self._hps.num_decoder_layers,
            mode=self._hps.mode,
            dropout=self._hps.dropout_rate,
            cell_wrapper=cell_wrappers.MultiRNNCell,
            cell_scope=self._scope.Decoder,
            # additional args
            state_is_tuple=True,
            initializer=self.rand_unif_init)

        # In decode mode, we run attention_decoder one step at a time and so
        # need to pass in the previous step's coverage vector each time
        prev_coverage = (self.prev_coverage
            if hps.mode == "decode" and hps.coverage else None)

        # tile start-tokens
        start_tokens = tf.contrib.seq2seq.tile_batch(tf.constant(
            [self._vocab.word2id(data.START_DECODING)]), hps.batch_size)


        (final_dists,
         out_state,
         attn_dists,
         p_gens,
         coverage,
         sampled_tokens,
         _, _, _) = pg_decoder.policy_gradient_pointer_attention_decoder(
            cell=cells,
            scope=self._scope,
            memory=self._enc_states,
            decoder_inputs=inputs,
            initial_state=self._dec_in_state,
            enc_padding_mask=self._enc_padding_mask,
            prev_coverage=prev_coverage,
            # tokens
            UNK_token=self._vocab.word2id(data.UNKNOWN_TOKEN),
            start_tokens=start_tokens,
            embeddings=embedding,
            vocab_size=self._vocab.size(),
            num_source_OOVs=self._max_art_oovs,
            enc_batch_extended_vocab=self._enc_batch_extend_vocab,
            # some flags
            reinforce=self._reinforce,
            pointer_gen=hps.pointer_gen,
            use_coverage=hps.coverage,
            # for decoding
            initial_state_attention=(hps.mode == "decode"))

        return (final_dists, out_state, attn_dists,
                p_gens, coverage, sampled_tokens)

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary

        # Some initializers
        self.rand_unif_init = tf.random_uniform_initializer(
            minval=-hps.rand_unif_init_mag,
            maxval=hps.rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(
            stddev=hps.trunc_norm_init_std)

        # Add embedding matrix (shared by the encoder and decoder inputs)
        with tf.variable_scope(self._scope.WordEmb):
            embedding = tf.get_variable(
                name="embedding", shape=[vsize, hps.emb_dim],
                dtype=tf.float32, initializer=self.trunc_norm_init)

            # tensor with shape (batch_size, max_enc_steps, emb_size)
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
            emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch)

        # Add the encoder.
        enc_outputs, enc_states = self._add_encoder(
            encoder_inputs=emb_enc_inputs,
            seq_len=self._enc_lens)
        
        self._enc_states = enc_outputs

        # Our encoder is bidirectional and our decoder is unidirectional so
        # we need to reduce the final encoder hidden state to the right
        # size to be the initial decoder hidden state
        self._dec_in_state = self._reduce_states(enc_states)

        # Add the decoder.
        (final_dists,
         self._dec_out_state,
         self.attn_dists,
         self.p_gens,
         self.coverage,
         self.sampled_tokens) = self._add_decoder(emb_dec_inputs, embedding)


        if hps.mode in ['train', 'eval']:
            # Calculate the loss
            with tf.variable_scope('loss'):
                if self._reinforce:
                    raise NotImplementedError("Not Tested")
                    # policy multipliers are the scaling term for its gradient
                    # for example: discounted rewards or advantage estimate
                    # here we use Generalized Advantage Estimation
                    self._policy_multipliers = tf.placeholder(
                        dtype=tf.float32, shape=self._target_batch.shape,
                        name="policy_multipliers")
                    nll_loss = pg_utils.negative_log_likelihood(
                        actions_prob=final_dists,
                        target_actions=self._sampled_tokens,
                        policy_multipliers=self._policy_multipliers,
                        episode_masks=self._dec_padding_mask,
                        action_space=self._vocab.size() + self._max_art_oovs)
                    self._summaries.append(tf.summary.scalar(
                        self._name + '/Rewards',
                        tf.reduce_mean(self._policy_multipliers)))
                else:
                    # just set them to ones
                    self._policy_multipliers = None
                    nll_loss = pg_utils.negative_log_likelihood(
                        actions_prob=final_dists,
                        target_actions=self._target_batch,
                        policy_multipliers=None,
                        episode_masks=self._dec_padding_mask,
                        action_space=self._vocab.size() + self._max_art_oovs)


                self._loss = nll_loss
                self._nll_loss = nll_loss

                self._reg_loss = None
                self._reg_pl_names_dict = None

                reg_loss, reg_pl_names_dict = self._add_soft_sharing()
                self._loss += reg_loss
                self._reg_loss = reg_loss
                self._reg_pl_names_dict = reg_pl_names_dict

                # Calculate coverage loss from the attention distributions
                if hps.coverage:
                    raise NotImplementedError
                    with tf.variable_scope('coverage_loss'):
                        self._coverage_loss = _coverage_loss(
                            attn_dists=self.attn_dists,
                            padding_mask=self._dec_padding_mask)
                        self._summaries.append(tf.summary.scalar(
                            self._name + '/coverage_loss',
                            self._coverage_loss))

                    self._total_loss = (self._loss +
                        hps.cov_loss_wt * self._coverage_loss)
                    self._summaries.append(tf.summary.scalar(
                        self._name + '/total_loss', self._total_loss))

                self._summaries.append(tf.summary.scalar(
                    self._name + '/loss', self._loss))

        if hps.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            # final_dists is a singleton list containing shape (batch_size,
            # extended_vsize)
            
            # assert len(final_dists) == 1
            assert_op = tf.Assert(
                tf.equal(tf.shape(final_dists)[1], 1),
                [tf.shape(final_dists)])
            # final_dists = final_dists[0]
            with tf.control_dependencies([assert_op]):
                final_dists = final_dists[:, 0, :]

            # take the k largest probs.
            # note batch_size=beam_size in decode mode
            topk_probs, self._topk_ids = tf.nn.top_k(
                final_dists, hps.batch_size * 2)
            self._topk_log_probs = tf.log(topk_probs + 1e-6)


    def _add_soft_sharing(self):
        if not self._soft_sharing_coef or self._soft_sharing_coef < 1e-6:
            raise ValueError("soft_sharing_coef too small")

        # when filtering vars, trainable_variables
        # will include previously built models
        # so filter them out
        filtering_fn = lambda name: (
            # make sure these are shared params
            name in self._soft_sharing_params and
            # make sure these are vars from this model
            self._scope.Model.name in name)
        
        (reg_loss,
         reg_pl_names_dict) = soft_utils.get_regularzation_loss(
            filtering_fn=filtering_fn,
            scope=self._scope,
            coef=self._soft_sharing_coef)
        
        return reg_loss, reg_pl_names_dict

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to
        # minimize
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(
            gradients, self._hps.max_grad_norm)

        # Add a summary
        self._summaries.append(tf.summary.scalar(self._name + '/global_norm', global_norm))

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self._hps.lr)
        
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self._add_seq2seq()
        if self._hps.mode == 'train':
            self._add_train_op()

        if len(self._summaries) > 0:
            self._summaries = tf.summary.merge(self._summaries)
        else:
            self._summaries = None

        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch,
                       reg_model_name=None,
                       reg_filtering_fn=None,
                       all_scopes=None):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss}

        # coverage
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        
        # soft sharing
        if not self._soft_sharing_coef:
            raise ValueError(
                "soft_sharing_coef cannot be 0 or None when turned on")
        if self._reg_loss is None or self._reg_pl_names_dict is None:
            raise ValueError(
                "reg_loss or reg_pl_names_dictcannot None when turned on")

        # calculate the regularization loss
        # and add them to the feed_dict
        feed_dict = soft_utils.calc_regularization_loss(
            filtering_fn=reg_filtering_fn,
            reg_pl_names_dict=self._reg_pl_names_dict,
            reg_model_name=reg_model_name,
            feed_dict=feed_dict,
            sess=sess,
            all_scopes=all_scopes)
        
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing
        summaries, nll_loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            # reg_loss is not included
            'nll_loss': self._nll_loss}

        if self._hps.coverage:
            raise NotImplementedError("See comments above")
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(
            batch, just_enc=True)  # feed the batch into the placeholders
        # run the encoder
        (enc_states, dec_in_states, global_step) = sess.run(
            [self._enc_states, self._dec_in_state, self.global_step], feed_dict)

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is
        # identical across the batch so we just take the top row.
        dec_in_state = [tf.nn.rnn_cell.LSTMStateTuple(dec_in_state.c[0],
                                                      dec_in_state.h[0])
                        for dec_in_state in dec_in_states]

        dec_in_state = cell_wrappers.to_MultiRNNLSTMStateTuple(dec_in_state)
        return enc_states, dec_in_state

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single
        # LSTMStateTuple for the batch
        new_dec_in_state = misc_utils.concate_multi_rnn_cell_states(
            dec_init_states,
            concat_fn=lambda x: np.concatenate(x, axis=0),
            expand_fn=lambda x: np.expand_dims(x, axis=0))

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if self._hps.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of
        # LSTMStateTuple -- one for each hypothesis
        new_states = misc_utils.split_multi_rnn_cell_states(results["states"])

        # Convert singleton list containing a tensor to a list of k arrays
        assert results['attn_dists'].shape[1] == 1
        attn_dists = results['attn_dists'][:, 0, :].tolist()

        if self._hps.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert results['p_gens'].shape[1] == 1
            p_gens = results['p_gens'][:, 0, :].tolist()
        else:
            p_gens = [None for _ in xrange(beam_size)]

        # Convert the coverage tensor to a list length k containing the
        # coverage vector for each hypothesis
        if self._hps.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in xrange(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step]
                       for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = sum(values_per_step) / dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(
        attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    # Coverage loss per decoder timestep. Will be list length max_dec_steps
    # containing shape (batch_size).
    covlosses = []
    for a in attn_dists:
        # calculate the coverage loss for this step
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss