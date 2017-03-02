from __future__ import division
import argparse
import numpy as np
import logging
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import subprocess
import sys
from vae import VAE
from data_utils import constants, data_utils
FLAGS = None

class config(object):
  seq_len = 40
  batch_size = 20
  n_hidden = 150
  n_latent = 30
  max_iter = 10000
  vocab_size = 14467
  dev_eval_iter = 500
  kl_weight_rate = 1/10


def plot_loss(kl_loss, xentropy):
  plt.clf()
  x = np.array(range(len(kl_loss)))
  plt.plot(x, kl_loss, 'r', label='KL loss')
  plt.plot(x, xentropy, 'b', label='Cross-entropy')
  plt.legend()
  if FLAGS.save_fig:
    plt.savefig('{}/loss.png'.format(FLAGS.save_fig), bbox_inches='tight')
  else:
    plt.show()


def save_model(sess, saver, name=''):
  if FLAGS.save_dir:
    fname = '{}/model.ckpt{}'.format(FLAGS.save_dir, name)
    saver.save(sess, fname)


def bleu_eval(vae, dev_feed, sess, saver, best_bleu):
  dev_out = sess.run(vae.test_output, dev_feed)
  save_batch(dev_out, constants.dev_out)
  cat = subprocess.Popen(("cat", constants.dev_out), stdout=subprocess.PIPE)
  try:
    multibleu = subprocess.check_output((constants.bleu_script, FLAGS.dev_idx), stdin=cat.stdout)
    logging.info("{}".format(multibleu))
    m = re.match("BLEU = ([\d.]+),", multibleu)
    new_bleu = float(m.group(1))
    if new_bleu > best_bleu:
      logging.info('Model achieves new best bleu')
      save_model(sess, saver, name='-dev_bleu')
    return new_bleu
  except Exception, e:
    logging.info("Multi-bleu error: {}".format(e))
    return 0.0

def save_batch(data, fname):
  if FLAGS.save_dir:
    fname = FLAGS.save_dir + fname
  with open(fname, 'w') as f_out:
    for out in data:
      out = out.astype(int).tolist()
      try:
        out = out[:out.index(constants.EOS_ID)]
      except ValueError:
        pass
      f_out.write('{}\n'.format(' '.join(map(str, out))))    
   

def read_data(f_idx, utils):
  seq_list = []
  seq_len_list = []
  with open(f_idx, 'r') as f_in:
    for line in f_in:
      seq = map(int, line.split())
      seq_len = len(seq) + 1 # account for EOS_ID
      if seq_len <= config.seq_len:
        seq_len_list.append(seq_len)
        seq_unk_replace = [w if w <= config.vocab_size else utils.UNK_ID for w in seq]
        seq_list.append(
          seq_unk_replace + [constants.EOS_ID] + (config.seq_len - seq_len) * [utils.PAD_ID])
  logging.info('Read {} sequences from {}'.format(len(seq_list), f_idx))
  return seq_list, seq_len_list


def get_batch(vae, seq_data, seq_lens, batch_idx=0):
  input_feed = {}
  enc_input_data = []
  seq_len = []
  loss_weights = np.ones((vae.seq_len + 1, vae.batch_size))
  for seq_id in range(batch_idx * vae.batch_size, (batch_idx + 1) * vae.batch_size):
    seq_id = seq_id % len(seq_data)
    enc_input_data.append(seq_data[seq_id])
    seq_len.append(seq_lens[seq_id])
    loss_weights[seq_lens[seq_id]:, len(enc_input_data) - 1] = 0

  enc_input_data = np.array(enc_input_data)
  dec_input_data = np.array([[constants.GO_ID] + list(seq) for seq in enc_input_data])
  target_data = [dec_in[1:] for dec_in in vae.dec_inputs]

  for l in range(vae.seq_len):
    input_feed[vae.enc_inputs[l].name] = enc_input_data[:, l]
  for l in range(vae.seq_len + 1):
    input_feed[vae.dec_inputs[l].name] = dec_input_data[:, l]
    input_feed[vae.loss_weights[l].name] = loss_weights[l, :]
  input_feed[vae.targets[-1].name] = np.zeros(vae.batch_size)
  input_feed[vae.sequence_lengths.name] = np.array(seq_len)
  return input_feed, enc_input_data

def get_utils():
  utils = data_utils()
  if FLAGS.no_zero_pad:
    utils.no_zero_pad()
  if FLAGS.swap_unk is not None:
    utils.swap_unk(FLAGS.swap_unk)
  
  return utils

def main(_):
  best_dev_bleu = 0.0
  utils = get_utils()
  train_seqs, train_seq_lens = read_data(FLAGS.train_idx, utils)
  dev_seqs, dev_seq_lens = read_data(FLAGS.dev_idx, utils)
  test_seqs, test_seq_lens = read_data(FLAGS.test_idx, utils)

  cell = tf.nn.rnn_cell.LSTMCell(config.n_hidden)
  vae = VAE(config.batch_size,
            config.seq_len,
            config.n_hidden,
            config.vocab_size,
            cell, config.n_latent, 
            annealing=FLAGS.annealing)
  loss_hist = {'kl': [], 'xentropy': []}
  saver = tf.train.Saver()
  dev_feed, _ = get_batch(vae, dev_seqs, dev_seq_lens)

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())

    for i in range(config.max_iter):
      if FLAGS.annealing:
        vae.kl_weight = 1 - np.exp(-i * config.kl_weight_rate)
      input_feed, _ = get_batch(vae, train_seqs, train_seq_lens, batch_idx=i)

      if FLAGS.plot_loss:
        _, kl_loss, xentropy_loss = sess.run([vae.train, vae.kl_loss, vae.xentropy_loss], input_feed)
        loss_hist['kl'].append(np.mean(kl_loss))
        loss_hist['xentropy'].append(np.mean(xentropy_loss))
      else:
        sess.run(vae.train, input_feed)
      
      if (i % config.dev_eval_iter == 0):
        loss_val = sess.run(vae.loss, input_feed)
        dev_bleu = bleu_eval(vae, dev_feed, sess, saver, best_dev_bleu)
        best_dev_bleu = max(dev_bleu, best_dev_bleu)
        logging.info("iter {}, train loss {}, dev BLEU {}".format((i+1), loss_val, dev_bleu))
        
    if FLAGS.plot_loss:
      plot_loss(loss_hist['kl'], loss_hist['xentropy'])

    input_feed, input_data = get_batch(vae, test_seqs, test_seq_lens)
    output = sess.run(vae.test_output, input_feed)
    for in_, out_ in zip(input_data, output):
      if constants.EOS_ID in out_:
        out_ = out_[:list(out_).index(constants.EOS_ID)]
      logging.info("Input: {}, Output: {}".format(in_, out_))
    '''
    logging.info("Generating")
    gen_out = vae.generate(sess)
    for out in gen_out:
      logging.info(out[:list(out).index(constants.EOS_ID)])
    '''
    save_model(sess, saver, name='final')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_idx', type=str, default=None,
                      help='Path to training data file')
  parser.add_argument('--dev_idx', type=str, default=None,
                      help='Path to validation data file')
  parser.add_argument('--test_idx', type=str, default=None,
                      help='Path to training data file')
  parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory for saving model')
  parser.add_argument('--load_model', type=str, default=None,
                      help='Directory from which to load model')
  parser.add_argument('--save_fig', type=str, default=None,
                      help='Location to save plots')
  parser.add_argument('--plot_loss', default=False, action='store_true',
                      help='Set if plotting encoder/decoder loss over time')
  parser.add_argument('--annealing', default=False, action='store_true',
                      help='Set if initially not weighting KL loss')
  parser.add_argument('--no_zero_pad', default=False, action='store_true',
                      help='Set if using UNK_ID=0 and PAD_ID=EOS_ID, instead of PAD_ID=0, UNK_ID=3')
  parser.add_argument('--swap_unk', default=None, type=int,
                      help='Set with value for UNK token to take (e.g. for premapped data)')
  parser.add_argument('--decoder_dropout', type=float, default=1.0, 
                      help='Proportion of input to replace with dummy value on decoder input')
  FLAGS = parser.parse_args()
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.set_random_seed(1234)
  np.random.seed(1234)
  tf.app.run()
