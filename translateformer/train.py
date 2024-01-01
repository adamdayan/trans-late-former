import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import ticker
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from .model import create_mask

def train(model, dataloader, criterion, optim):
    loss_total = 0
    cnt = 0
    model.train()
    with tqdm(dataloader, unit="batch") as batch_iter:
      for src, targ, labels, src_mask, targ_mask in batch_iter:
          # forward pass
          logits = model(src, targ, src_mask, targ_mask)

          # cross entropy loss expects the "unrolled" logits (i.e on a per token-position basis)
          output_dim = logits.shape[-1]
          logits = logits.contiguous().view(-1, output_dim)
          labels = labels.contiguous().view(-1)

          loss = criterion(logits, labels)
          loss_total += loss.item()
          cnt += 1

          # backward pass
          optim.zero_grad()
          loss.backward()
          # we clip the gradients because I saw a few examples which did it and empirically I've found it gave me better results  -\_o_/-
          nn.utils.clip_grad_norm_(model.parameters(), 1)

          optim.step()

      return loss_total /cnt



def evaluate(model, dataloader, criterion):
  loss = 0
  cnt = 0
  model.eval()
  with torch.no_grad():
      with tqdm(dataloader, unit="batch") as batch_iter:
          for src, targ, labels, src_mask, targ_mask in batch_iter:
              logits = model(src, targ, src_mask, targ_mask)

              output_dim = logits.shape[-1]
              logits = logits.contiguous().view(-1, output_dim)
              labels = labels.contiguous().view(-1)

              loss += criterion(logits, labels).item()

              cnt += 1

  model.train()
  return loss / cnt


def greedy_decoding(model, src, context_size, sos_idx, eos_idx, pad_idx, device):
  cur_targ = torch.tensor([[sos_idx]])
  src_mask, _ = create_mask(src.to("cpu"), src.to("cpu"), pad_idx)
  features = model.encode(src.to(device), src_mask.to(device))
  model.eval()
  for i in range(context_size):
      src_mask, targ_mask = create_mask(src, cur_targ.to("cpu"), pad_idx)
      proj = model.decode(cur_targ.to(device), features.to(device), src_mask.to(device), targ_mask.to(device))
      tok = torch.argmax(proj[0,-1,:], keepdims=True)
      tok = tok.unsqueeze(0)

      if tok.item() == eos_idx:
          return cur_targ[:,1:]
      cur_targ = torch.cat([cur_targ.to(device), tok], dim=1).type(torch.int)
  model.train()
  return cur_targ

def calculate_bleu(model, dataloader, context_size, sos_idx, pad_idx, eos_idx, vocab, device):
  pred_sentences = []
  label_sentences = []
  model.eval()
  with torch.no_grad():
    with tqdm(dataloader, unit="batch") as batch_iter:
      # vectorise greedy decoding for better performance by computing translations of whole batch at a time 
      for src, targ, labels, src_mask, targ_mask in batch_iter:
        # create dummy inputs at start of sentence
        cur_targ_raw = [[sos_idx] for _ in range(labels.shape[0])]
        translation_done = [False for _ in range(labels.shape[0])]
        features = model.encode(src, src_mask)

        for i in range(context_size):
          cur_targ = torch.LongTensor(cur_targ_raw)
          src_mask, targ_mask = create_mask(src, cur_targ.to("cpu"), pad_idx)
          proj = model.decode(cur_targ.to(device), features, src_mask.to(device), targ_mask.to(device))
          toks = torch.argmax(proj[:, -1,:], dim=-1, keepdims=True)

          for ti in range(toks.shape[0]):
            if toks[ti] in [eos_idx, pad_idx]:
              translation_done[ti] = True
            cur_targ_raw[ti].append(toks[ti].item())
          if all(translation_done):
            break

        for targ_sentence in cur_targ_raw:
          pred_sentence = []
          for i in range(1, len(targ_sentence)):
              if targ_sentence[i] == eos_idx:
                  break
              pred_sentence.append(vocab.textify(targ_sentence[i]))
          pred_sentences.append(pred_sentence)

        label_sentences.extend([[vocab.textify(tok) for tok in label if tok not in [eos_idx, pad_idx]]] for label in labels.tolist())

  print(pred_sentences)
  print(label_sentences)
  model.train()
  return bleu_score(pred_sentences, label_sentences)

def visualise_attention(attention_matrix, src_sentence, targ_sentence, save_path=None):
  attention_matrix = attention_matrix.cpu().detach()
  n_heads = attention_matrix.shape[1]
  fig, axs = plt.subplots(2, 4, figsize=(15, 25))
  for i, ax in enumerate(axs.flat):
     ax.tick_params(labelsize=12)
     ax.matshow(attention_matrix[0, i, :,:])
     ax.set_yticklabels(targ_sentence)
     ax.set_xticklabels(src_sentence)
     ax.label_outer()
     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  if save_path is not None:
     plt.savefig(save_path)