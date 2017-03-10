class data_utils(object):
  def __init__(self):
    self.PAD_ID = 0
    self.UNK_ID = 3

  def no_zero_pad(self):
    self.PAD_ID = constants.EOS_ID
    self.UNK_ID = 0
  
  def swap_unk(self, num):
    self.UNK_ID = num

class constants(object):
  GO_ID = 1
  EOS_ID = 2
  bleu_script = "/home/mifs/ds636/code/scripts/multi-bleu.perl"
