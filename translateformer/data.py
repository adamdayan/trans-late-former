import torch
from torch.utils.data import DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from .model import create_mask

def get_unique_tokens(text, tokenizer, min_freq):
    cnt = Counter()
    for doc in tokenizer.pipe(text):
      doc_tokens = [tok.text.strip().lower() for tok in doc if len(tok.text) > 0]
      cnt.update(doc_tokens)
    return [tok for tok, freq in cnt.most_common() if freq >= min_freq]

class Vocab:
  # special token indices
  UNK_IDX = 0
  PAD_IDX = 1
  SOS_IDX = 2
  EOS_IDX = 3

  UNK = '<unk>' # Unknown
  PAD = '<pad>' # Padding
  SOS = '<sos>' # Start of sentence
  EOS = '<eos>' # End of sentence

  SPECIAL_TOKENS = [UNK, PAD, SOS, EOS]

  def __init__(self, tokenizer, tokens):
    self.tokenizer = tokenizer
    self.tokens = self.SPECIAL_TOKENS + tokens
    self.token_idxs = {self.tokens[i]: i for i in range(len(self.tokens))}

  def __call__(self, text):
      return [self.numerify(tok.text.strip().lower()) for tok in self.tokenizer(text)]

  def numerify(self, tok):
    if tok not in self.token_idxs:
      return self.UNK_IDX
    return self.token_idxs[tok]

  def textify(self, idx):
    return self.tokens[idx]

  def to_string(self, idxs):
    for i in range(idxs.shape[0]):
      return " ".join([self.textify(idx) for idx in idxs[i]])

def make_dataloader(dataset, batch_size, context_size, src_vocab, targ_vocab, device):
  def sort_by_length(bucket):
    return sorted(bucket, key= lambda x: len(x[0]))

  # TODO: test effect of this
  # dataset = dataset.bucketbatch(batch_size=batch_size, sort_key=sort_by_length, bucket_num=1)

  def collate_fn(batch):
    srcs = []
    targs = []

    for i, (src_sentence, targ_sentence) in enumerate(batch):
      if len(src_sentence) == 0 or len(targ_sentence) == 0:
        continue
      src = [src_vocab.SOS_IDX] + src_vocab(src_sentence) + [src_vocab.EOS_IDX]
      targ = [targ_vocab.SOS_IDX] + targ_vocab(targ_sentence) + [targ_vocab.EOS_IDX]

      srcs.append(torch.tensor(src))
      targs.append(torch.tensor(targ))

    src_batch = pad_sequence(srcs, padding_value=src_vocab.PAD_IDX, batch_first=True)
    targ_batch = pad_sequence(targs, padding_value=targ_vocab.PAD_IDX, batch_first=True)

    # decoder wants target starting with SOS
    target_batch = targ_batch[:, :-1]
    # however when calculating loss we are only interested in tokens after SOS and ending with EOS
    label_batch = targ_batch[:, 1:]
    src_mask, targ_mask = create_mask(src_batch, target_batch, src_vocab.PAD_IDX) 

    all_batches = [src_batch, target_batch, label_batch, src_mask, targ_mask]
    return [b.to(device, dtype=torch.long) for b in all_batches]

  return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
