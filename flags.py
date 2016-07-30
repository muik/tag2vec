import os
import subprocess
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS

class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data

    self.interactive = FLAGS.interactive

  @classmethod
  def tag(cls):
    opts = Options()
    opts.train_data = 'data/tags.txt'
    opts.save_path = 'train'
    opts.eval_data = 'data/questions-tags.txt'
    opts.window_size = 5
    opts.min_count = 7

    with open(os.devnull, 'w') as FNULL:
      if subprocess.call(['ls', 'train'], stdout=FNULL) != 0:
        subprocess.call(['wget', 'https://muik-projects.firebaseapp.com/tf/tag2vec-train.tgz'],
            stdout=FNULL)
        subprocess.call(['tar', 'xvfz', 'tag2vec-train.tgz'])
        subprocess.call(['rm', 'tag2vec-train.tgz'])
    return opts
