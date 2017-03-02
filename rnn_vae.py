from __future__ import division
import argparse
import numpy as np
import logging
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import subprocess
from vae import VAE
FLAGS = None

class constants(object):
  PAD_ID = 0
  GO_ID = 1
  EOS_ID = 2
  dev_out = 'dev_out'
  bleu_script = "/home/mifs/ds636/code/scripts/multi-bleu.perl"

class config(object):
  seq_len = 40
  batch_size = 20
  n_hidden = 30
  n_latent = 10
  max_iter = 500
  dev_eval_iter = 100
  kl_weight_rate = 1/10



def get_batch(vae, seqs_seq_lens):
  input_feed = {}
  enc_input_data = []
  seq_len = []
  loss_weights = np.ones((vae.seq_len + 1, vae.batch_size))
  for seq in range(vae.batch_size):
    enc_input_data.append(np.pad(r, (0, vae.seq_len - rand_len), 'constant'))
    seq_len.append(rand_len)
    loss_weights[rand_len:, seq] = 0

  enc_input_data = np.array(enc_input_data)
  dec_input_data = np.array([[constants.GO_ID] + list(seq) for seq in enc_input_data])
  target_data = [dec_in[1:] for dec_in in vae.dec_inputs]

  for l in range(vae.seq_len):
    input_feed[vae.enc_inputs[l].name] = enc_input_data[:, l]
  for l in range(vae.seq_len + 1):
    input_feed[vae.dec_inputs[l].name] = dec_input_data[:, l]
    input_feed[vae.loss_weights[l].name] = loss_weights[l, :]
  input_feed[vae.targets[-1].name] = np.zeros(vae.batch_size)
  input_feed[vae.seq_len.name] = np.array(seq_len)
  return input_feed, enc_input_data


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
    fname = '{}/model.ckpt{}'.format(name)
    saver.save(sess, FLAGS.save_dir + '/model.ckpt')

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
  with open(fname, 'w') as f_out:
    for out in data:
      out = out.astype(int).tolist()
      try:
        out = out[:out.index(constants.EOS_ID)]
      except ValueError:
        pass
      f_out.write('{}\n'.format(' '.join(map(str, out))))    
   
def read_data(f_idx):
  seq_list = []
  seq_len_list = []
  with open(f_idx, 'r') as f_in:
    for line in f_in:
      seq = map(int, line.split())
      seq_len = len(seq) + 1 # account for EOS_ID
      if seq_len <= config.seq_len:
        seq_len_list.append(seq_len)
        seq_list.append(seq + [constants.EOS_ID] + (config.seq_len - seq_len) * [constants.PAD_ID])
  logging.info('Read {} sequences from {}'.format(len(seq_list), f_idx))
  return zip(seq_list, seq_len_list)

def main(_):
  best_dev_bleu = 0.0
  train_seqs = read_data(FLAGS.train_idx)
  dev_seqs = read_data(FLAGS.dev_idx)
  test_seqs = read_data(FLAGS.test_idx)

  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  vae = VAE(config.batch_size,
            config.seq_len,
            config.n_hidden,
            config.vocab_size,
            cell, config.n_latent, 
            annealing=FLAGS.annealing)
  loss_hist = {'kl': [], 'xentropy': []}
  saver = tf.train.Saver()
  dev_feed, _ = get_batch(vae, dev_seqs)

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())

    for i in range(config.max_iter):
      if FLAGS.annealing:
        vae.kl_weight = 1 - np.exp(-i * config.kl_weight_rate)
      input_feed, _ = get_batch(vae, train_seqs)

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

    input_feed, input_data = get_batch(vae, test_seqs)
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
    save_model(sess, saver)


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
  parser.add_argument('--decoder_dropout', type=float, default=1.0, 
                      help='Proportion of input to replace with dummy value on decoder input')
  FLAGS = parser.parse_args()
  tf.set_random_seed(1234)
  np.random.seed(1234)
  tf.app.run()
