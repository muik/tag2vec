# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import random

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec
from flags import FLAGS, Options

class Word2Vec(object):
  """Word2Vec model (Skipgram)."""
  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    if not options.interactive:
      self.save_vocab()
      self._read_analogies()
      self._load_corpus()

  def _read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.decode('utf-8').strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def get_no_words(self, words):
    return [word for word in words if word not in self._word2id]

  def get_vocab_size(self):
    return self._options.vocab_size

  def get_emb_dim(self):
    return self._options.emb_dim

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    if opts.interactive:
      self._id2word = np.loadtxt(os.path.join(opts.save_path, "vocab.txt"), 'str', unpack=True)[0]
      self._id2word = [x.decode('utf-8') for x in self._id2word]
      for i, w in enumerate(self._id2word):
        self._word2id[w] = i
      opts.vocab_size = len(self._id2word)
      self._w_in = tf.get_variable('w_in', [opts.vocab_size, opts.emb_dim])
      return

    # The training data. A text file.
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=opts.train_data,
                                           batch_size=opts.batch_size,
                                           window_size=opts.window_size,
                                           min_count=opts.min_count,
                                           subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    opts.vocab_words = map(lambda x: x.decode('utf-8'), opts.vocab_words)
    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    examples = tf.placeholder(dtype=tf.int32)  # [N]
    labels = tf.placeholder(dtype=tf.int32)  # [N]

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train(w_in,
                                 w_out,
                                 examples,
                                 labels,
                                 lr,
                                 vocab_count=opts.vocab_counts.tolist(),
                                 num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]).encode('utf-8'),
                             opts.vocab_counts[i]))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    word_ids = tf.placeholder(dtype=tf.int32)  # [N]
    negative_word_ids = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    words_emb = tf.nn.embedding_lookup(nemb, word_ids)
    negative_words_emb = tf.nn.embedding_lookup(nemb, negative_word_ids)

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)
    self._target = target
    self._dist = dist

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    mean = tf.reduce_mean(words_emb, 0)
    mean = tf.reshape(mean, [-1, opts.emb_dim])
    mean_dist = 1.0 - tf.matmul(mean, words_emb, transpose_b=True)
    _, self._mean_pred_idx = tf.nn.top_k(mean_dist, 1)

    joint_dist = tf.matmul(words_emb, nemb, transpose_b=True)
    n_joint_dist = tf.matmul(negative_words_emb, nemb, transpose_b=True)
    joint_dist = tf.reduce_sum(joint_dist, 0) - tf.reduce_sum(n_joint_dist, 0)
    self._joint_idx = tf.nn.top_k(joint_dist, min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._word_ids = word_ids
    self._negative_word_ids = negative_word_ids
    self._analogy_pred_idx = pred_idx

    self.saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(opts.save_path))
    if ckpt:
      self.saver.restore(self._session, ckpt)
      print('loaded %s' % ckpt)
    else:
      # Properly initialize all variables.
      tf.initialize_all_variables().run()

  def _load_corpus(self):
    corpus = []
    with open(self._options.train_data, 'r') as f:
      unk_id = self._word2id['UNK']
      def word2id(w):
        return w in self._word2id and self._word2id[w] or unk_id
      while True:
        line = f.readline().decode('utf-8')
        if not line:
          break
        corpus.append([word2id(w) for w in line.split()])
    self._corpus = corpus
    self._corpus_lines_count = len(corpus)

  def _batch_data(self):
    examples = []
    labels = []
    batch_size = self._options.batch_size
    window_size = self._options.window_size
    unk_id = self._word2id['UNK']
    count = 0
    while True:
      line = self._corpus[random.randrange(0,self._corpus_lines_count)]
      words_count = len(line)
      for i, center_id in enumerate(line):
        if center_id == unk_id:
          continue
        outputs = line[max(0, i-window_size):i] + line[min(words_count-1, i+1):min(words_count-1, i+1+window_size)]
        outputs = [x for x in outputs if x != unk_id and x != center_id]
        outputs_count = len(outputs)
        examples += [center_id] * outputs_count
        labels += outputs
        count += outputs_count
        if count >= batch_size:
          return examples[:batch_size], labels[:batch_size]

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      examples, labels = self._batch_data()
      _, epoch = self._session.run([self._train, self._epoch], {
          self._examples: examples,
          self._labels: labels
      })
      if epoch != initial_epoch:
        break
#      time.sleep(0.02) # for preventing notebook noise

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(2)  # Reports our progress once a while.
      (epoch, step, words,
       lr) = self._session.run([self._epoch, self.step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    total = self._analogy_questions.shape[0]
    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    accuracy = correct * 100.0 / total
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, accuracy))
    return accuracy

  def get_nearby(self, words, negative_words, num=20):
    wids = [self._word2id.get(w, 0) for w in words]
    n_wids = [self._word2id.get(w, 0) for w in negative_words]
    idx = self._session.run(self._joint_idx, {
        self._word_ids: wids,
        self._negative_word_ids: n_wids
    })
    results = []
    for distance, i in zip(idx[0][:num], idx[1][:num]):
      if i in wids:
        continue
      results.append((self._id2word[i], distance))
    return results

  def doesnt_match(self, *words):
    wids = [self._word2id.get(w, 0) for w in words]
    idx, = self._session.run(self._mean_pred_idx, {
        self._word_ids: wids
    })
    print(words[idx[0]])
    return

  def get_doesnt_match(self, *words):
    wids = [self._word2id.get(w, 0) for w in words]
    idx, = self._session.run(self._mean_pred_idx, {
        self._word_ids: wids
    })
    return words[idx[0]]

  def get_analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2, 'UNK']:
        return c
    return

  def save(self):
    opts = self._options
    self.saver.save(self._session, os.path.join(opts.save_path, "model.ckpt"))
    self.export_tsne()
    print('Saved')

  def export_tsne(self):
    from sklearn.manifold import TSNE
    import json
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    with self._session.as_default():
      low_dim_embs = tsne.fit_transform(self._w_in.eval()[1:plot_only,:])
    labels = [self._id2word[i] for i in xrange(plot_only)]
    embs = [list(e) for e in low_dim_embs]
    json_data = json.dumps({'embs': embs, 'labels': labels})
    with open(os.path.join(self._options.save_path, 'tsne.js'), 'w') as f:
      f.write(json_data)


def main(_):
  """Train a word2vec model."""
  opts = Options.tag()
  if not opts.train_data or not opts.save_path or not opts.eval_data:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)

  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
    for i in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      accuracy = model.eval()  # Eval analogies.
      if (i+1) % 5 == 0:
        model.save()
    if opts.epochs_to_train % 5 != 0:
      model.save()


class Tag2vec:
  def __init__(self):
    opts = Options.tag()
    opts.interactive = True
    session = tf.Session()
    self.model = Word2Vec(opts, session)


if __name__ == "__main__":
  tf.app.run()
