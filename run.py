from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import tensorflow as tf
from datetime import datetime
from collections import namedtuple

from pointer_model.data import Vocab
from pointer_model.decode import BeamSearchDecoder
from pointer_model.model import SummarizationModel

from utils import misc_utils
from evaluation_utils.evaluators import evaluate
from multitask.multitask_base_model import MultitaskBatcher
from multitask.multitask_base_model import MultitaskBaseModel
from multitask.multitask_autoMR_model import MultitaskAutoMRModel


tf.logging.set_verbosity(tf.logging.INFO)
NAMES = "Newsela,SNLI,PP"
StepsPerVal = 10
StepsPerCheckpoint = 1500
AutoMRNumValBatches = 2
AutoMRStepsPerUpdate = 10
ValNLL_Normalizing_Constant = 2
MultitaskBatcherArgs = namedtuple("MultitaskBatcherArgs",
    ("data_paths", "vocabs", "hps", "single_pass"))
HParamsList = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag',
               'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim',
               'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
               'coverage', 'cov_loss_wt', 'pointer_gen',
               # additionals
               "num_encoder_layers", "num_decoder_layers", "dropout_rate"]


def add_arguments():
    parser = argparse.ArgumentParser()
    # Hparams (default are good)
    parser.add_argument("--steps_per_eval",
                        type=int, default=1500,
                        help="number of steps for evaluation")
    parser.add_argument("--hidden_dim",
                        type=int, default=256,
                        help="dimension of RNN hidden states")
    parser.add_argument("--emb_dim",
                        type=int, default=128,
                        help="dimension of word embeddings")
    parser.add_argument("--batch_size",
                        type=int, default=256,
                        help="minibatch size")
    parser.add_argument("--max_enc_steps",
                        type=int, default=None,
                        help="max timesteps of encoder")
    parser.add_argument("--max_dec_steps",
                        type=int, default=None,
                        help="max timesteps of decoder")
    parser.add_argument("--min_dec_steps",
                        type=int, default=1,
                        help="Minimum sequence length of generated summary. "
                             "Applies only for beam search decoding mode")
    parser.add_argument("--vocab_size",
                        type=int, default=50000,
                        help="Size of vocabulary")
    parser.add_argument("--rand_unif_init_mag",
                        type=float, default=0.02,
                        help="magnitude for lstm cells random uniform inititalization")
    parser.add_argument("--trunc_norm_init_std",
                        type=float, default=1e-4,
                        help="std of trunc norm init, used for initializing everything else")
    parser.add_argument("--max_grad_norm",
                        type=float, default=2.0,
                        help="for gradient clipping")
    parser.add_argument("--pointer_gen",
                        type=bool, default=True,
                        help="Use pointer-generator model")
    parser.add_argument("--coverage",
                        type=bool, default=False,
                        help="Use coverage mechanism.")
    parser.add_argument("--convert_to_coverage_model",
                        type=bool, default=False,
                        help="Convert a non-coverage model to a coverage model.")
    parser.add_argument("--cov_loss_wt",
                        type=float, default=1.0,
                        help="Weight of coverage loss (lambda in the paper)"
                             "If zero, then no incentive to minimize coverage loss.")
    
    # Hyparams need to change
    parser.add_argument("--num_encoder_layers",
                        type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_decoder_layers",
                        type=int, default=2,
                        help="number of layers")
    parser.add_argument("--dropout_rate",
                        type=float, default=None,
                        help="dropout_rate = 1 - keep_prob")
    parser.add_argument("--lr",
                        type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--beam_size",
                        type=int, default=None,
                        help="beam size for beam search decoding.")
    parser.add_argument("--max_hours",
                        type=int, default=None,
                        help="number of hours before killing the model.")

    # model directories
    parser.add_argument("--mode",
                        type=str, default=None,
                        help="train or decode")

    parser.add_argument("--log_root",
                        type=str, default=None,
                        help="Root directory for all logging.")

    parser.add_argument("--exp_name",
                        type=str, default=None,
                        help="Name for experiment. Logs will be saved "
                             "in a directory with this name, under log_root.")
    
    parser.add_argument("--vocab_path",
                        type=str, default=None,
                        help="path to vocabulary")
    
    parser.add_argument("--train_data_dirs",
                        type=str, default=None,
                        help="Comma-separated: "
                             "path expression to tf.Example datafiles. ")
    
    parser.add_argument("--val_data_dir",
                        type=str, default=None,
                        help="path expression to tf.Example datafiles. ")

    # evaluation
    parser.add_argument("--eval_source_dir",
                        type=str, default=None,
                        help="Directory to the evaluation source")

    parser.add_argument("--eval_target_dir",
                        type=str, default=None,
                        help="Directory to the evaluation target")
    
    parser.add_argument("--eval_folder_dir",
                        type=str, default=None,
                        help="directory to the evaluation folder")
    
    # load models
    parser.add_argument("--load_ckpt_file",
                        type=str, default=None,
                        help="restore from specific checkpints")

    # decoding
    parser.add_argument("--decode_data_dir",
                        type=str, default=None,
                        help="directory to the file for decoding")

    parser.add_argument("--decode_ckpt_file",
                        type=str, default=None,
                        help="checkpoint files for decoding only")

    parser.add_argument("--decode_output_file",
                        type=str, default=None,
                        help="outputs of decoding")

    parser.add_argument("--names",
                        type=str, default=NAMES)
    parser.add_argument("--mixing_ratios",
                        type=str, default=None)
    parser.add_argument("--soft_sharing_coef",
                        type=float, default=None)
    parser.add_argument("--autoMR",
                        action="store_true", default=False)
    parser.add_argument("--reward_scaling_factor",
                        type=float, default=ValNLL_Normalizing_Constant,
                        help="reward scaling")
    parser.add_argument("--selector_alpha",
                        type=float, default=0.3)

    

    FLAGS, unparsed = parser.parse_known_args()
    
    # convert comma-separated strings into lists
    FLAGS.names = FLAGS.names.split(",")
    FLAGS.mixing_ratios = (
        [float(x) for x in FLAGS.mixing_ratios.split(",")]
        if FLAGS.mixing_ratios is not None else None)
    
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)
    
    if FLAGS.train_data_dirs is None:
        if FLAGS.mode != "decode":
            raise ValueError("train_data_dirs cannot be None")
        # else keep it None, since it doesnt matter
    else:
        # check compatability
        FLAGS.train_data_dirs = FLAGS.train_data_dirs.split(",")
        if not len(FLAGS.names) == len(FLAGS.train_data_dirs):
            raise ValueError("names and train_data_dirs not match")

    if (FLAGS.mixing_ratios is not None and
            len(FLAGS.names) != len(FLAGS.mixing_ratios) + 1):
        raise ValueError("names and mixing_ratios + 1 not match")

    if not FLAGS.soft_sharing_coef or FLAGS.soft_sharing_coef < 1e-6:
        raise ValueError("not really supported")


    if FLAGS.dropout_rate is not None:
        raise ValueError("Not supporting dropout")

    # Make a namedtuple hps
    hps_dict = {}
    for key, val in vars(FLAGS).items():
        if key in HParamsList:
            hps_dict[key] = val
    hps = namedtuple("HParams", hps_dict.keys())(** hps_dict)
    return FLAGS, hps


def _model_factory(name):
    model = SummarizationModel
    print("Task %s is using %s" % (name, model.__name__))
    return model


def setup_training(FLAGS, hps):
    """Does setup before starting training (run_training)"""
    
    # Setting up the Multitask Wrapper
    # ----------------------------------------
    if FLAGS.autoMR:
        # for decode, we can still use this one
        # since both are essentially the same
        # except no auto-MR feature
        MultitaskModel = MultitaskAutoMRModel
    else:
        MultitaskModel = MultitaskBaseModel

    # Setting up the models and directories
    # ----------------------------------------
    num_models = len(FLAGS.names)
    # train_dir is a folder, decode_dir is a file
    train_dir = os.path.join(FLAGS.log_root, "train")
    decode_dir = os.path.join(FLAGS.log_root, "decode")
    model_creators = [_model_factory(name) for name in FLAGS.names]
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Setting up the batchers and data readers
    # ----------------------------------------
    print("Loading Training Data from %s " % FLAGS.train_data_dirs)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    train_batchers = MultitaskBatcher(
        data_paths=FLAGS.train_data_dirs,
        vocabs=[vocab for _ in range(num_models)],
        hps=hps, single_pass=False)
    # not using decode_model_hps which have batch-size = beam-size
    val_batchers = MultitaskBatcher(
        data_paths=[FLAGS.val_data_dir],
        vocabs=[vocab], hps=hps, single_pass=False)

    # Setting up the task selectors
    # ----------------------------------------
    Q_initial = -1
    if FLAGS.reward_scaling_factor > 0.0:
        Q_initial = Q_initial / FLAGS.reward_scaling_factor
        tf.logging.info("Normalization %.2f" % FLAGS.reward_scaling_factor)

    # Build
    # ----------------------------------------
    print("Mixing ratios are %s " % FLAGS.mixing_ratios)
    train_models = MultitaskModel(
        names=FLAGS.names,
        all_hparams=[hps for _ in range(num_models)],
        mixing_ratios=FLAGS.mixing_ratios,
        model_creators=model_creators,
        logdir=train_dir,
        soft_sharing_coef=FLAGS.soft_sharing_coef,
        data_generators=train_batchers,
        val_data_generator=val_batchers,
        vocab=vocab,
        selector_Q_initial=Q_initial,
        alpha=FLAGS.selector_alpha,
        temperature_anneal_rate=None)

    # Note this use a different decoder_batcher
    
    # The model is configured with max_dec_steps=1 because we only ever run
    # one step of the decoder at a time (to do beam search). Note that the
    # batcher is initialized with max_dec_steps equal to e.g. 100 because
    # the batches need to contain the full summaries

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need
    # to make a batch of these hypotheses.
    decode_model_hps = hps
    decode_model_hps = hps._replace(
        mode="decode")._replace(batch_size=FLAGS.beam_size)

    # we need to constantly re-initialize this generator
    # so save arguments as a namedtuple
    print("Loading Validation Data from %s " % FLAGS.val_data_dir)
    decode_batcher_args = MultitaskBatcherArgs(
        data_paths=[FLAGS.val_data_dir],
        vocabs=[vocab],
        hps=decode_model_hps,
        single_pass=True)
    
    decode_batchers = (
        MultitaskBatcher(** decode_batcher_args._asdict()))

    # only for one model
    decode_models = MultitaskBaseModel(
        names=[FLAGS.names[0]],
        all_hparams=[decode_model_hps._replace(max_dec_steps=1)],
        mixing_ratios=None,
        model_creators=[model_creators[0]],
        logdir=train_dir,
        soft_sharing_coef=FLAGS.soft_sharing_coef,
        vocab=vocab)

    with decode_models.graph.as_default():
        decoder = BeamSearchDecoder(model=decode_models,
                                    batcher=decode_batchers,
                                    vocab=vocab,
                                    ckpt_dir=train_dir,
                                    decode_dir=decode_dir,
                                    FLAGS=FLAGS)
        decode_sess = tf.Session(graph=decode_models.graph,
                                 config=misc_utils.get_config())
        decoder.build_graph(decode_sess)

    try:
        # this is an infinite loop until interrupted
        run_training(FLAGS=FLAGS,
                     models=train_models,
                     decoder=decoder,
                     decode_batcher_args=decode_batcher_args)
    
    except KeyboardInterrupt:
        tf.logging.info("Stopped...")


def run_training(FLAGS, models, decoder, decode_batcher_args):
    tf.logging.info("Initializing ...")
    models.initialize_or_restore_session(ckpt_file=FLAGS.load_ckpt_file)
    
    start_time = datetime.now()
    tf.logging.info("Starting run_training at %s, will run "
                    "for %s hours", start_time, FLAGS.max_hours)

    while True:
        with misc_utils.calculate_time("seconds for training step"):
            models.run_train_step()

        elapsed_hours = (datetime.now() - start_time).seconds // 3600
        if FLAGS.max_hours and elapsed_hours >= FLAGS.max_hours:
            models.save_session()
            break

        # update the val-loss as Q-values
        # define Q as negative val-loss
        if FLAGS.autoMR and models.global_step % AutoMRStepsPerUpdate == 0:
            total_val_loss = 0
            for _ in range(AutoMRNumValBatches):
                val_loss = models.run_eval_step()
                total_val_loss += val_loss

            # Q = negaative average val-loss
            scores = -float(total_val_loss) / float(AutoMRNumValBatches)
            # reward scaling
            if FLAGS.reward_scaling_factor > 0.0:
                scores = scores / float(FLAGS.reward_scaling_factor)
            # update the Q values
            models.update_TaskSelector_Q_values(scores)

        if models.global_step % StepsPerCheckpoint == 0:
            models.save_session()

        if (models.global_step != 0 and
                models.global_step % FLAGS.steps_per_eval == 0):
            # save checkpoints
            models.save_session()
            # run decode for calculating scores
            decoder.decode()
            # reset batcher from exhausted state
            decode_batchers = (
                MultitaskBatcher(** decode_batcher_args._asdict()))
            decoder.reset_batcher(decode_batchers)
            # evaluate generated outputs and log results

            scores = evaluate(
                mode="val",
                gen_file=decoder._decode_dir,
                ref_file=FLAGS.eval_target_dir,
                execute_dir=FLAGS.eval_folder_dir,
                source_file=FLAGS.eval_source_dir,
                evaluation_task=FLAGS.names[0])

            print(scores)


def setup_and_run_decoding(FLAGS, hps):
    # raise ValueError("Pay attention to dropout is set or not")
    if os.path.exists(FLAGS.decode_output_file):
        raise ValueError("`decode_output_file` exists")

    decode_model_hps = hps
    decode_model_hps = hps._replace(
        mode="decode")._replace(batch_size=FLAGS.beam_size)
    train_dir = os.path.join(FLAGS.log_root, "train")
    model_creators = [_model_factory(name) for name in FLAGS.names]

    print("Loading Decoding Data from %s " % FLAGS.decode_data_dir)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    decode_batchers = MultitaskBatcher(
        data_paths=[FLAGS.decode_data_dir],
        vocabs=[vocab],
        hps=decode_model_hps,
        single_pass=True)

    # only for one model
    decode_models = MultitaskBaseModel(
        names=[FLAGS.names[0]],
        all_hparams=[decode_model_hps._replace(max_dec_steps=1)],
        mixing_ratios=None,
        model_creators=[model_creators[0]],
        logdir=train_dir,
        soft_sharing_coef=FLAGS.soft_sharing_coef,
        # additional args
        vocab=vocab)

    with decode_models.graph.as_default():
        decoder = BeamSearchDecoder(model=decode_models,
                                    batcher=decode_batchers,
                                    vocab=vocab,
                                    ckpt_dir=train_dir,
                                    decode_dir=FLAGS.decode_output_file,
                                    FLAGS=FLAGS)
        decode_sess = tf.Session(graph=decode_models.graph,
                                 config=misc_utils.get_config())
        decoder.build_graph(decode_sess)

        # run decode for calculating scores
        decoder.decode(ckpt_file=FLAGS.decode_ckpt_file)

    scores = evaluate(
        mode="test",
        gen_file=decoder._decode_dir,
        ref_file=FLAGS.eval_target_dir,
        execute_dir=FLAGS.eval_folder_dir,
        source_file=FLAGS.eval_source_dir,
        evaluation_task=FLAGS.names[0])

    print(scores)


def main(unused_argv):
    tf.set_random_seed(111)
    FLAGS, hps = add_arguments()

    if hps.mode == 'train':
        print("creating training model...")
        setup_training(FLAGS, hps)

    elif hps.mode == 'decode':
        print("creating decoding model")
        setup_and_run_decoding(FLAGS, hps)

    else:
        raise ValueError(hps.mode)


if __name__ == '__main__':
    tf.app.run()
