from __future__ import division
import tensorflow as tf
import numpy as np
import logging
from data_utils import constants
from tensorflow.python.ops import rnn


class VAE(object):
  def __init__(self, batch_size, seq_len, n_hidden, embedding_size, vocab_size, cell, n_latent, 
               initializer=tf.truncated_normal, transfer_func=tf.nn.relu, annealing=False):
    self.batch_size = batch_size
    self.seq_len = seq_len 
    self.vocab_size = vocab_size
    self.n_hidden = n_hidden
    self.embedding_size = embedding_size
    self.n_latent = n_latent
    self.annealing = annealing
    self.kl_weight = tf.placeholder(tf.float32, shape=[], name="klw")
    self.z_gen = tf.placeholder(tf.float32, shape=[None, self.n_latent], name="zgen")
    self.enc_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{}".format(i))
                       for i in range(self.seq_len)]
    self.dec_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{}".format(i))
                       for i in range(self.seq_len + 1)]
    self.loss_weights = [tf.placeholder(tf.float32, shape=[None], name="weight{}".format(i))
                       for i in range(self.seq_len + 1)]
    self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="seqlen")
    self.targets = [self.dec_inputs[i + 1] for i in range(len(self.dec_inputs) - 1)]
    self.targets.append(tf.placeholder(tf.int32, shape=[None], name="last_target"))
    self.initializer = initializer
    self.transfer_func = transfer_func
    self._enc_cell = cell
    self._dec_cell = cell
    self.build_network()

  def get_latent(self, enc_state):
    z_mean_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent],
                                            dtype=tf.float32))
    z_mean_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))
    z_logvar_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent],
                                              dtype=tf.float32))
    z_logvar_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))

    self.z_mean = self.transfer_func(tf.add(tf.matmul(enc_state, z_mean_w), z_mean_b))
    self.z_log_var = self.transfer_func(tf.add(tf.matmul(enc_state, z_logvar_w), z_logvar_b))
    eps = tf.random_normal(tf.shape(self.z_log_var), 0, 1, dtype=tf.float32)

    self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))

  def build_network(self):
    def get_out(to_reshape):
      reshaped = tf.reshape(tf.concat(1, to_reshape), [-1, self.n_hidden])
      return tf.reshape(
        tf.argmax(tf.nn.softmax(tf.matmul(reshaped, softmax_w) + softmax_b), 1),
        [-1, self.seq_len + 1])

    with tf.variable_scope('encoder'):
      embedding = tf.Variable(self.initializer([self.vocab_size, self.embedding_size], dtype=tf.float32))
      embedded_inputs = [tf.nn.embedding_lookup(embedding, enc_input) 
                         for enc_input in self.enc_inputs]
      _, enc_state = tf.nn.rnn(self._enc_cell, embedded_inputs, dtype=tf.float32,
                               sequence_length=self.sequence_lengths)

    self.get_latent(enc_state)
    dec_in_w = tf.Variable(self.initializer([self.n_latent, self._dec_cell.state_size],
                                              dtype=tf.float32))
    dec_in_b = tf.Variable(tf.zeros([self._dec_cell.state_size], dtype=tf.float32))
    dec_initial_state = self.transfer_func(tf.add(tf.matmul(self.z, dec_in_w), dec_in_b))
    dec_gen_initial_state = self.transfer_func(tf.add(tf.matmul(self.z_gen, dec_in_w), dec_in_b))

    with tf.variable_scope('decoder') as scope:
      softmax_w = tf.Variable(self.initializer([self.n_hidden, self.vocab_size]))
      softmax_b = tf.Variable(tf.zeros([self.vocab_size]))
      output_projection=[softmax_w, softmax_b]
      dec_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_initial_state,
                                                        self._dec_cell, self.vocab_size,
                                                        self.n_hidden, output_projection,
                                                        feed_previous=False)
      scope.reuse_variables()
      dec_test_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_initial_state,
                                                            self._dec_cell, self.vocab_size,
                                                            self.n_hidden, output_projection,
                                                            feed_previous=True,
                                                             update_embedding_for_previous=False)

      dec_gen_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_gen_initial_state,
                                                             self._dec_cell, self.vocab_size,
                                                             self.n_hidden, output_projection,
                                                             feed_previous=True,
                                                             update_embedding_for_previous=False)

      self.logits = [tf.add(tf.matmul(dec_out, softmax_w), softmax_b) 
                     for dec_out in dec_outs]
      self.output = [tf.argmax(tf.nn.softmax(logit), 1) for logit in self.logits]
      
      self.train_output = get_out(dec_outs)
      self.gen_output = get_out(dec_gen_outs)
      self.test_output = get_out(dec_test_outs)
    seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
        self.logits,
        self.targets,
        self.loss_weights)
    self.xentropy_loss = tf.reduce_sum(seq_loss) / self.batch_size
    self.kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                        - tf.square(self.z_mean)
                                        - tf.exp(self.z_log_var), 1)
    if self.annealing:
      self.loss = tf.reduce_mean(self.kl_weight * self.kl_loss + self.xentropy_loss)
    else:
      self.loss = tf.reduce_mean(self.kl_loss + self.xentropy_loss)
    self.train = tf.train.AdamOptimizer().minimize(self.loss)

      

  def generate(self, sess, z_mu=None, batch_size=None):
    logging.info("Generating")
    if batch_size is not None:
      orig_batch_size, self.batch_size = self.batch_size, batch_size
    if z_mu is None:
      z_mu = np.random.normal(size=(self.batch_size, self.n_latent))

    input_feed = {self.z_gen: z_mu}
    input_feed[self.dec_inputs[0].name] = constants.GO_ID * np.ones(self.batch_size)
    gen_out = sess.run(self.gen_output, input_feed)
    for out in gen_out:
      logging.info(out[:list(out).index(constants.EOS_ID)])
    if batch_size is not None:
      self.batch_size = origi_batch_size
