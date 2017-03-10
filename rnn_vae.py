from __future__ import division
import numpy as np
import logging
import matplotlib 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import re
import subprocess
import sys
from vae import VAE
from data_utils import constants, data_utils

tf.app.flags.DEFINE_string("config_file", None, "Pass options in a config file (overrides cmdline)")

tf.app.flags.DEFINE_string("train_idx", None, "Path to training data file")
tf.app.flags.DEFINE_string("dev_idx", None, "Path to validation data file")
tf.app.flags.DEFINE_string("test_idx", None, "Path to test data file")
tf.app.flags.DEFINE_string("train_dir", None, "Directory to save model files")
tf.app.flags.DEFINE_string("load_model", None, "Directory from which to load model")
tf.app.flags.DEFINE_boolean("plot_loss", False, "True if plotting encoder/decoder loss over steps")
tf.app.flags.DEFINE_boolean("annealing", False, "Set if initially unweighting KL loss")
tf.app.flags.DEFINE_boolean("no_zero_pad", False, "Use UNK=0 and PAD=EOS, not default PAD=0, UNK=3")
tf.app.flags.DEFINE_integer("swap_unk", None, "Value for UNK token (e.g. for premapped data)") 
tf.app.flags.DEFINE_integer("embedding_size", 350, "Size of input embedding") 
tf.app.flags.DEFINE_integer("n_latent", 15, "Size of latent layer") 
tf.app.flags.DEFINE_integer("n_hidden", 200, "Hidden layer size for LSTM cell") 
tf.app.flags.DEFINE_integer("vocab_size", 10000, "Vocab size for input/output data") 
tf.app.flags.DEFINE_integer("dev_eval_iter", 200, "Iterations between evaluating dev data") 
tf.app.flags.DEFINE_integer("batch_size", 20, "Training examples per iteration") 
tf.app.flags.DEFINE_integer("seq_len", 50, "Maximum length of training sequence") 
tf.app.flags.DEFINE_integer("max_iter", 10000, "Max number of training iterations") 
tf.app.flags.DEFINE_float("decoder_dropout", 1.0, "Proportion of input to replace by UNK at decoder")
tf.app.flags.DEFINE_float("kl_weight_rate", 0.01, "Weight for KL loss annealing")

FLAGS = tf.app.flags.FLAGS
config = {}

def plot_loss(kl_loss, xentropy):
  plt.clf()
  x = np.array(range(len(kl_loss)))
  plt.plot(x, kl_loss, 'r', label='KL loss')
  plt.plot(x, xentropy, 'b', label='Cross-entropy')
  plt.legend()
  plt.savefig('{}/loss.png'.format(config['train_dir']), bbox_inches='tight')
  plt.show()

def save_model(sess, saver, name=''):
  fname = '{}/model.ckpt-{}'.format(config['train_dir'], name)
  saver.save(sess, fname)

def bleu_eval(vae, sess, saver, best_bleu):
  cat = subprocess.Popen(("cat", config['dev_out']), stdout=subprocess.PIPE)
  try:
    multibleu = subprocess.check_output((constants.bleu_script, config['dev_idx']), stdin=cat.stdout)
    logging.info("{}".format(multibleu))
    m = re.match("BLEU = ([\d.]+),", multibleu)
    new_bleu = float(m.group(1))
    if new_bleu > best_bleu:
      logging.info('Model achieves new best bleu')
      save_model(sess, saver, name='dev_bleu')
    return new_bleu
  except Exception, e:
    logging.info("Multi-bleu error: {}".format(e))
    return 0.0

def save_data(data, fname):
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
      if seq_len <= config['seq_len']:
        seq_len_list.append(seq_len)
        seq_unk_replace = [w if w < config['vocab_size'] else utils.UNK_ID for w in seq]
        seq_list.append(
          seq_unk_replace + [constants.EOS_ID] + (config['seq_len'] - seq_len) * [utils.PAD_ID])
  logging.info('Read {} sequences from {}'.format(len(seq_list), f_idx))
  return seq_list, seq_len_list

def output_test(vae, sess, seq_data, seq_lens, batch_size=20, eval_count=0, fname=None):
  orig_batch_size, vae.batch_size = vae.batch_size, batch_size
  out_seqs = []
  for i in range(int(len(seq_data) / batch_size)):
    input_feed, input_data = get_batch(vae, seq_data, seq_lens, batch_idx=i)
    output = sess.run(vae.train_output, input_feed)
    for in_, out_ in zip(input_data, output):
      if constants.EOS_ID in in_:
        in_ = in_[:list(in_).index(constants.EOS_ID)]
      if constants.EOS_ID in out_:
        out_ = out_[:list(out_).index(constants.EOS_ID)]
      out_seqs.append(out_)
      logging.info("Input: {}, Output: {}".format(in_, out_))
      if eval_count > 0 and i == eval_count:
        break
  if fname is not None:
    save_data(out_seqs, fname)
  vae.batch_size = orig_batch_size


def get_batch(vae, seq_data, seq_lens, batch_idx=0, train=True):
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
  if config['annealing']:
    input_feed[vae.kl_weight.name] = 1 - np.exp(-batch_idx * config['kl_weight_rate'])
  input_feed[vae.targets[-1].name] = np.zeros(vae.batch_size)
  input_feed[vae.sequence_lengths.name] = np.array(seq_len)
  return input_feed, enc_input_data

def get_utils():
  utils = data_utils()
  if config['no_zero_pad']:
    utils.no_zero_pad()
  if config['swap_unk'] is not None:
    utils.swap_unk(config['swap_unk'])
  logging.info('Using UNK_ID={} and PAD_ID={}'.format(utils.UNK_ID, utils.PAD_ID))
  return utils

def process_args():
  # Override flags with config file options if necessary
  def is_bool(v):
    return v.lower() in ('true', 't', 'false', 'f')
  def str2bool(v):
    return v.lower() in ('true', 't')  
  for key, value in FLAGS.__dict__['__flags'].iteritems():
    config[key] = value
  cfg = FLAGS.config_file
  if cfg is not None:
    if not cfg or not os.path.isfile(cfg):
      raise ValueError("Cannot load config file %s" % cfg)
    else:
      logging.info('Reading settings from config file')
      with open(cfg, 'r') as f:
        for line in f:
           key, value = line.strip().split(": ")
           if value.lower() == "none":
             value = None
           elif is_bool(value):
             value = str2bool(value)      
           elif re.match("^\d+$", value):
             value = int(value)
           elif re.match("^[\d\.]+$", value):
             value = float(value)
           config[key] = value
  config['dev_out'] = '{}/dev_out'.format(config['train_dir'])

  for key in config:
    logging.info('{}: {}'.format(key, config[key]))
           

def main(_):
  process_args()
  best_dev_bleu = 0.0
  utils = get_utils()
  train_seqs, train_seq_lens = read_data(config['train_idx'], utils)
  dev_seqs, dev_seq_lens = read_data(config['dev_idx'], utils)
  test_seqs, test_seq_lens = read_data(config['test_idx'], utils)
  
  cell = tf.nn.rnn_cell.LSTMCell(config['n_hidden'])
  vae = VAE(config['batch_size'], config['seq_len'], config['n_hidden'], config['embedding_size'], config['vocab_size'], cell, config['n_latent'], annealing=config['annealing'])
  loss_hist = {'kl': [], 'xentropy': []}
  saver = tf.train.Saver()
  with tf.Session() as sess:
    if config['load_model']:
      saver.restore(sess, config['load_model'])
    else:
      sess.run(tf.initialize_all_variables())

    for i in range(config['max_iter']):
      input_feed, _ = get_batch(vae, train_seqs, train_seq_lens, batch_idx=i)

      if config['plot_loss']:
        _, kl_loss, xentropy_loss = sess.run([vae.train, vae.kl_loss, vae.xentropy_loss], input_feed)
        loss_hist['kl'].append(np.mean(kl_loss))
        loss_hist['xentropy'].append(np.mean(xentropy_loss))
      else:
        sess.run(vae.train, input_feed)
      
      if (i % config['dev_eval_iter'] == 0):
        loss_val, kl_loss, xentropy_loss = sess.run([vae.loss, vae.kl_loss, vae.xentropy_loss],
                                                    input_feed)
        output_test(vae, sess, dev_seqs, dev_seq_lens, batch_size=vae.batch_size,
                    fname=config['dev_out'])
        dev_bleu = bleu_eval(vae, sess, saver, best_dev_bleu)
        best_dev_bleu = max(dev_bleu, best_dev_bleu)
        logging.info("iter {}, train loss {}, KL loss {}, cross-entropy loss: {}, dev BLEU {}".format(
          (i+1), loss_val, np.mean(kl_loss), np.mean(xentropy_loss), dev_bleu))
        
    if config['plot_loss']:
      plot_loss(loss_hist['kl'], loss_hist['xentropy'])

    output_test(vae, sess, test_seqs, test_seq_lens, batch_size=1)
    # vae.generate(sess, batch_size=10)
    save_model(sess, saver, name='final')


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.set_random_seed(1234)
  np.random.seed(1234)
  tf.app.run()
