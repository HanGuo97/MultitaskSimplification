import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

from TFLibrary.SPG import pg_decoder
from TFLibrary.utils import test_utils
from TFLibrary.utils import attention_utils
_zero_state_tensors = rnn_cell_impl._zero_state_tensors

print("Using Tensorflow ", tf.__version__)


# Set Up ########################################
class Scope(object):
    Decoder = test_utils.create_scope("decoder")
    Attention = test_utils.create_scope("attention")
    Pointer = test_utils.create_scope("pointer")
    Projection = test_utils.create_scope("projection")
    

class HPS(object):
    batch_size = 32
    enc_seq_length = 20
    dec_seq_length = 10
    embedding_size = 256
    num_units = 256
    vocab_size = 100
    OOVs = 100


class FakeVocab(object):
    def size(self):
        return HPS.vocab_size


def test():
    _test_seq2seq_attention_pointer()


def _test_seq2seq_attention_pointer():
    # Set Up
    # ==================================================
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HPS.num_units) for _ in range(2)])
    encoder_states = test_utils.random_tensor(
        [HPS.batch_size, HPS.enc_seq_length, HPS.num_units])
    decoder_inputs = test_utils.random_tensor(
        [HPS.batch_size, HPS.dec_seq_length, HPS.embedding_size])
    initial_state = cell.zero_state(HPS.batch_size, tf.float32)
    enc_padding_mask = test_utils.random_integers(
        low=0, high=2, dtype=tf.float32,
        shape=[HPS.batch_size, HPS.enc_seq_length])
    start_tokens = test_utils.random_integers(
        low=0, high=HPS.vocab_size + HPS.OOVs, shape=[HPS.batch_size])
    embeddings = test_utils.random_tensor(
        [HPS.vocab_size, HPS.embedding_size])
    enc_batch_extended_vocab = test_utils.random_integers(
        low=0, high=(HPS.vocab_size + HPS.OOVs),
        shape=[HPS.batch_size, HPS.enc_seq_length])


    # Test Seq2Seq with Attention
    # =======================================================

    # Actual Outputs
    # -------------------------------------------------------
    (final_dists, final_cell_state, attn_dists, p_gens,
     coverage, sampled_tokens, decoder_outputs_ta,
     debug_variables, final_loop_state) = (
        pg_decoder.policy_gradient_pointer_attention_decoder(
            cell=cell,
            scope=Scope,
            memory=encoder_states,
            decoder_inputs=decoder_inputs,
            initial_state=initial_state,
            enc_padding_mask=enc_padding_mask,
            # token
            UNK_token=0,
            start_tokens=start_tokens,
            embeddings=embeddings,
            vocab_size=HPS.vocab_size,
            num_source_OOVs=HPS.OOVs,
            enc_batch_extended_vocab=enc_batch_extended_vocab,
            # flags
            reinforce=False))

    # cell_input is cell inputs
    (sampled_tokens_history,
     outputs_history, alignments_history, p_gens_history,
     coverages_history, logits_history, vocab_dists_history,
     final_dists_history, coverage, cell_input) = final_loop_state

    # Expected Outputs
    # -------------------------------------------------------
    (all_cell_output,
     all_next_cell_state,
     all_cell_inputs,
     all_attention,
     all_context,
     all_alignments) = _attention_rnn_cell(
        cell=cell,
        scope=Scope,
        num_units=HPS.num_units,
        batch_size=HPS.batch_size,
        inputs=decoder_inputs,
        memory=encoder_states,
        mask=enc_padding_mask,
        query_layer=debug_variables["query_kernel"],
        memory_layer=debug_variables["memory_kernel"],
        input_layer=debug_variables["input_kernel"],
        attention_layer=debug_variables["output_kernel"],
        initial_cell_state=initial_state)


    # Check differences
    # -------------------------------------------------------
    sess = tf.Session()
    sess.__enter__()
    tf.global_variables_initializer().run()

    cell_outputs_diff = (
        tf.stack(all_attention) - decoder_outputs_ta.stack())
    last_cell_state_diff = (
        tf.stack(all_next_cell_state[-1]) - tf.stack(final_cell_state))
    alignments_diff = (
        tf.transpose(all_alignments, perm=[1, 0, 2]) - tf.stack(attn_dists))

    test_utils.tensor_is_zero(sess, cell_outputs_diff, "CellOutputsDiff")
    test_utils.tensor_is_zero(sess, last_cell_state_diff, "LastCellStateDiff")
    test_utils.tensor_is_zero(sess, alignments_diff, "AlignmentsDiff")


    # Test p_gens
    # =======================================================

    def _stack_and_transpose(X):
        X = array_ops.stack(X)
        X = array_ops.transpose(X, perm=[1, 0, 2])
        return X
        
    def _calculate_p_gens(contexts, cell_states, cell_inputs, pgen_layer):
        p_gens = array_ops.concat([
            _stack_and_transpose(contexts),
            _stack_and_transpose([s[-1].c for s in cell_states]),
            _stack_and_transpose([s[-1].h for s in cell_states]),
            _stack_and_transpose(cell_inputs)], axis=-1)
        p_gens = pgen_layer(p_gens)
        return p_gens

    # Expected Outputs
    # -------------------------------------------------------
    pgens = _calculate_p_gens(
        all_context,
        all_next_cell_state,
        all_cell_inputs,
        debug_variables["pgen_kernel"])
    
    pgens_diff = pgens - p_gens
    test_utils.tensor_is_zero(sess, pgens_diff, "PgensDIff")



    # Test Vocab Distribution
    # =======================================================

    # Expected Outputs
    # -------------------------------------------------------
    all_attn_dists = all_alignments
    all_logits = map(lambda X:
        debug_variables["logits_kernel"](X), all_attention)
    all_vocab_dists = map(lambda X: tf.nn.softmax(X), all_logits)
    all_vocab_dists = list(all_vocab_dists)
    diff_vocab_dists = tf.stack(all_vocab_dists) - vocab_dists_history.stack()
    test_utils.tensor_is_zero(sess, diff_vocab_dists, "DiffVocabDists")

    # Test Final Distribution
    # =======================================================

    # Expected Outputs
    # -------------------------------------------------------
    fake_self = test_utils.DictClass({
        "p_gens": tf.unstack(pgens, axis=1),
        "_vocab": FakeVocab(),
        "_max_art_oovs": HPS.OOVs,
        "_hps": test_utils.DictClass({"batch_size": HPS.batch_size}),
        "_enc_batch_extend_vocab": enc_batch_extended_vocab})
    # print("self.p_gens ", fake_self.p_gens)
    # print("self._vocab.size() ", fake_self._vocab.size())
    # print("self._max_art_oovs ", fake_self._max_art_oovs)
    # print("self._hps.batch_size ", fake_self._hps.batch_size)
    # print("self._enc_batch_extend_vocab ", fake_self._enc_batch_extend_vocab)

    expected_final_dist = _calc_final_dist(
        fake_self, all_vocab_dists, all_attn_dists)
    final_dist_diff = _stack_and_transpose(expected_final_dist) - final_dists
    test_utils.tensor_is_zero(sess, final_dist_diff, "DiffFinalDists")


    # Test pg_decoder._calc_final_dist
    # -------------------------------------------------------
    actual_final_dist2 = [
        pg_decoder._calc_final_dist(
            vocab_dist=vd,
            attn_dist=ad,
            p_gen=pg,
            batch_size=fake_self._hps.batch_size,
            vocab_size=fake_self._vocab.size(),
            num_source_OOVs=fake_self._max_art_oovs,
            enc_batch_extended_vocab=fake_self._enc_batch_extend_vocab)
        
        for vd, ad, pg in zip(all_vocab_dists,
                              all_attn_dists,
                              fake_self.p_gens)]

    final_dist_diff2 = (_stack_and_transpose(expected_final_dist) -
                        _stack_and_transpose(actual_final_dist2))
    test_utils.tensor_is_zero(sess, final_dist_diff2, "DiffFinalDists")



def _attention_cell(cell,
                   scope,
                   input_layer,
                   attention_layer,
                   attention_mechanism,
                   inputs, cell_state, attention):
    """More flexible attention wrapper"""
    with vs.variable_scope(scope, reuse=True):
        # compute cell inputs
        cell_inputs = input_layer(
            array_ops.concat([inputs, attention], -1))

    with vs.variable_scope("cell", reuse=True):
        # run cell
        cell_output, next_cell_state = cell(cell_inputs, cell_state)
    

    with vs.variable_scope(scope, reuse=True):
        # Computes attention and alignments
        alignments, _ = attention_mechanism(cell_output, state=None)
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        context = math_ops.matmul(expanded_alignments,
                                  attention_mechanism.values)
        context = array_ops.squeeze(context, [1])

        # compute attention output
        if attention_layer is not None:
            attention = attention_layer(
                array_ops.concat([cell_output, context], 1))
        else:
            attention = context


    return (cell_output,  # direct outputs from cell
            next_cell_state,  # next cell state
            cell_inputs,  # attention concatenated cell inputs
            attention,  # linear projection of [cell_outputs; context]
            context,  # convext vector
            alignments)  # attention distribution


def _attention_rnn_cell(cell,
                        scope,
                        num_units,
                        batch_size,
                        inputs,
                        memory,
                        mask,
                        query_layer,
                        memory_layer,
                        input_layer,
                        attention_layer,
                        initial_cell_state):
    attention_mechanism = attention_utils.BahdanauAttentionTester(
        num_units=num_units,
        memory=memory,
        mask=mask,
        query_layer=query_layer,
        memory_layer=memory_layer,
        scope=scope.Attention)
    
    next_cell_state = initial_cell_state
    context = _zero_state_tensors(
        num_units, batch_size, tf.float32)
    sequence_length = inputs.get_shape()[1].value
    
    all_cell_output = []
    all_next_cell_state = []
    all_cell_inputs = []
    all_attention = []
    all_context = []
    all_alignments = []
    
    for time in range(sequence_length):
        (cell_output,
         next_cell_state,
         cell_inputs,
         attention,
         context,
         alignments) = _attention_cell(cell=cell,
                                      scope=scope.Attention,
                                      input_layer=input_layer,
                                      attention_layer=attention_layer,
                                      attention_mechanism=attention_mechanism,
                                      inputs=inputs[:, time, :],
                                      cell_state=next_cell_state,
                                      attention=context)
        all_cell_output.append(cell_output)
        all_next_cell_state.append(next_cell_state)
        all_cell_inputs.append(cell_inputs)
        all_attention.append(attention)
        all_context.append(context)
        all_alignments.append(alignments)
    
    return (all_cell_output,
            all_next_cell_state,
            all_cell_inputs,
            all_attention,
            all_context,
            all_alignments)


def _calc_final_dist(self, vocab_dists, attn_dists):
    with tf.variable_scope("Projection"):
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = [
            p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist)
                      in zip(self.p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the
        # probabilities for in-article OOV words
        # the maximum (over the batch) size of the extended vocabulary
        extended_vsize = self._vocab.size() + self._max_art_oovs
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        # list length max_dec_steps of shape (batch_size, extended_vsize)
        vocab_dists_extended = [
            tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        # shape (batch_size)
        batch_nums = tf.range(0, limit=self._hps.batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = tf.shape(self._enc_batch_extend_vocab)[
            1]  # number of states we attend over
        # shape (batch_size, attn_len)
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        # shape (batch_size, enc_t, 2)
        indices = tf.stack(
            (batch_nums, self._enc_batch_extend_vocab), axis=2)
        shape = [self._hps.batch_size, extended_vsize]
        # list length max_dec_steps (batch_size, extended_vsize)
        attn_dists_projected = [tf.scatter_nd(
            indices, copy_dist, shape) for copy_dist in attn_dists]

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a
        # [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(
            vocab_dists_extended, attn_dists_projected)]

        return final_dists


if __name__ == "__main__":
    test()
