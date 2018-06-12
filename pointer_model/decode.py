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
import os
import tensorflow as tf
import data
import pyrouge
import logging
import warnings
from pointer_model import beam_search

from utils import misc_utils

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab, ckpt_dir, decode_dir, FLAGS):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
          ckpt_dir: directory to checkpoints
          decode_dir: directory to decoding outputs
        """
        self._model = model
        self._batcher = batcher
        self._vocab = vocab
        self._ckpt_dir = ckpt_dir
        self._decode_dir = decode_dir
        self._FLAGS = FLAGS
        

    def reset_batcher(self, batcher):
        self._batcher = batcher

    def build_graph(self, sess):
        self._sess = sess
        self._saver = tf.train.Saver()

    def decode(self, ckpt_file=None):
        FLAGS = self._FLAGS

        # load latest checkpoint
        misc_utils.load_ckpt(self._saver, self._sess, self._ckpt_dir, ckpt_file)

        counter = 0
        f = open(self._decode_dir, "w")
        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                tf.logging.info(
                    "Decoder has finished reading dataset for single_pass.")
                break

            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[
                0]  # list of strings

            article_withunks = data.show_art_oovs(
                original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[
                                                   0] if FLAGS.pointer_gen else None))  # string

            # Run beam search to get best Hypothesis
            best_hyp = beam_search.run_beam_search(
                self._sess, self._model, self._vocab, batch, FLAGS)

            # Extract the output ids from the hypothesis and convert back to
            # words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(
                output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                # index of the (first) [STOP] symbol
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            # write ref summary and decoded summary to file, to eval with
            # pyrouge later
            # self.write_for_rouge(original_abstract_sents, decoded_words, counter)
            processed = self.depreciated_processing(decoded_words)
            f.write(processed + "\n")
            
            counter += 1
            if counter % 100 == 0:
                print("%d sentences decoded" % counter)

        f.close()


    def depreciated_processing(self, decoded_words):
        """Depreciated function, will be removed later"""
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                # fst_period_idx = decoded_words.index(".")
                fst_period_idx = len(decoded_words)
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            # sentence up to and including the period
            sent = decoded_words[:fst_period_idx + 1]
            decoded_words = decoded_words[
                fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        if not len(decoded_sents) == 1:
            warnings.warn("Found multiple decoded_sents in "
                          "`depreciated_processing`: ", decoded_sents)
            #raise ValueError(
            #    "Found multiple decoded_sents in `depreciated_processing`: ",
            #    decoded_sents)
        return decoded_sents[0]


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(
        logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (
                key, val, val_cb, val_ce)
    tf.logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    tf.logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)


def correct_unk(sentences):
    new_sentences = []
    for sent in sentences:
        new_sent = sent.replace("<unk>", "UNK").replace("unk", "UNK")
        new_sentences.append(new_sent)
    return new_sentences


