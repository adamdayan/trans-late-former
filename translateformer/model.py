import portalocker
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def create_mask(src, targ, pad_token):
  # NOTE: not fully confident in the unsqueeze - i think we get the extra-dimension so the masked_fill can broadcast it
  # we want to broadcast for the padding mask because it's the same on both axes (unlike with the causal mask)
  src_pad_mask = (src != pad_token).unsqueeze(1).unsqueeze(1) # (B, 1, 1, context_size)
  full_mask = torch.ones((targ.shape[1], targ.shape[1])).type(torch.int).unsqueeze(0).unsqueeze(0) # (1, 1, targ_context_size, targ_context_size)
  causal_atten_mask = torch.tril(full_mask) # (1, 1, targ_context_size, targ_context_size)
  targ_pad_mask = (targ != pad_token).unsqueeze(1).unsqueeze(1)

  return src_pad_mask, targ_pad_mask & causal_atten_mask

def initialize_weights(p):
    if hasattr(p, 'weight') and p.weight.dim() > 1:
        nn.init.xavier_uniform_(p.weight.data)

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, n_embed, dropout, store_attention):
      super().__init__()
      self.n_embed = n_embed
      self.n_heads = n_heads
      assert(n_embed % n_heads == 0) # check dims work
      self.head_size = n_embed // n_heads
      self.dropout = dropout

      self.wk = nn.Linear(n_embed, n_embed, bias=True)
      self.wq = nn.Linear(n_embed, n_embed, bias=True)
      self.wv = nn.Linear(n_embed, n_embed, bias=True)
      self.proj = nn.Linear(n_embed, n_embed)

      self.dropout = nn.Dropout(dropout)

      self.store_attention = store_attention

  def forward(self, x, features, mask):
      B, T, C = features.shape # (batch, context_size, n_embed)

      # create k
      k = self.wk(features) # (B, context_size, n_embed) @ (n_embed, n_embed) ---> (B, context_size, n_embed)
      # split per head
      k = k.view(B, T, self.n_heads, self.head_size)  # (B, context_size, n_embed) --> (B, context_size, n_heads, head_size)
      # switch context size and n_heads dim so we can batch matmul over B and n_heads
      k = k.transpose(1,2) # (B, context_size, n_heads, head_size) --> (B, n_heads, context_size, head_size)

      # create q
      q = self.wq(x)
      q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_size) # q can be a different size to k and v so need to use its own .shape
      q = q.transpose(1,2)

      # create v
      v = self.wv(features)
      v = v.view(B, T, self.n_heads, self.head_size)
      v = v.transpose(1, 2)

      attn = q @ k.transpose(-2, -1) # (B, n_heads, context_size, head_size) @ (B, n_heads, head_size, context_size) --> (B, n_heads, context_size, context_size)
      attn = attn / (q.shape[-1] ** 0.5) # divide by squareroot of n_embed to decrease magnitude

      # if this is a masked attention layer (causal?) mask out all tokens before the cur pos, otherwise just mask out padding
      attn = attn.masked_fill(mask == 0, -1e10) # TODO: check that the dimensions here work
      attn = self.dropout(F.softmax(attn, dim=-1))

      if self.store_attention:
         self.attn_store = attn

      # generate the v matrix and use the attn matrix to pluck out relevant info on a per head basis
      out = attn @ v # (B, n_heads, context_size, context_size) @ (B, n_heads, context_size, head_size) --> (B, n_heads, context_size, head_size)

      # remove per-head dimension and use final linear projection
      out = out.transpose(1,2).contiguous()
      out = out.view(out.shape[0], -1, self.n_embed) # (B, context_size , n_head, head_size) --> (B, context_size, n_embed)
      out = self.proj(out)
      return out

class FeedForward(nn.Module):
  def __init__(self, n_embed, ff_expansion_factor, dropout):
      super().__init__()
      self.ffw = nn.Sequential(
          nn.Linear(n_embed, ff_expansion_factor* n_embed),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(ff_expansion_factor*n_embed, n_embed),
      )

  def forward(self, x):
      return self.ffw(x) # (B, context_size, n_embed) @ (n_embed, ff_expansion_factor*n_embed) @ (ff_expansion_factor*n_embed, n_embed) --> (B, context_size, n_embed)

class EncoderBlock(nn.Module):
  def __init__(self, n_heads, n_embed, ff_expansion_factor, dropout, store_attention):
      super().__init__()
      self.attention = MultiHeadAttention(n_heads, n_embed, dropout, store_attention)
      self.ffw = FeedForward(n_embed, ff_expansion_factor, dropout)
      self.ln1 = nn.LayerNorm(n_embed)
      self.ln2 = nn.LayerNorm(n_embed)
      self.dropout = nn.Dropout(dropout)

  def forward(self, inputs):
      x, mask = inputs
      out = self.ln1(
        self.dropout(self.attention(x, x, mask)) + x
      )
      out = self.ln2(
        self.dropout(self.ffw(out)) + out
      )
      return (out, mask)

class DecoderBlock(nn.Module):
  def __init__(self, n_heads, n_embed, ff_expansion_factor, dropout, store_attention):
      super().__init__()
      self.ln1 = nn.LayerNorm(n_embed)
      self.ln2 = nn.LayerNorm(n_embed)
      self.ln3 = nn.LayerNorm(n_embed)

      self.masked_attention = MultiHeadAttention(n_heads, n_embed, dropout, store_attention)
      self.cross_attention = MultiHeadAttention(n_heads, n_embed, dropout, store_attention)
      self.ffw = FeedForward(n_embed, ff_expansion_factor, dropout)

      self.dropout = nn.Dropout(dropout)

  def forward(self, inputs):
      y, features, src_mask, targ_mask = inputs

      out = self.ln1(
          self.dropout(self.masked_attention(y, y, targ_mask)) + y
      )

      out = self.ln2(
          self.dropout(self.cross_attention(out, features, src_mask)) + out
      )
      out = self.ln3(
          self.dropout(self.ffw(out)) + out
      )

      return (out, features, src_mask, targ_mask)

class PositionalEmbedding(nn.Module):
  def __init__(self, n_embed, context_size, device):
      super().__init__()
      self.posem = nn.Embedding(context_size, n_embed)
      self._device = device

  def forward(self, x):
      pos = torch.arange(0, x.shape[1]).unsqueeze(0).to(self._device)
      return x + self.posem(pos)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, context_size, dropout=0.1):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)

      # sinusoidal positional encoding as in "Attention is All You Need"
      pe = torch.zeros(context_size, d_model)
      position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0)
      self.register_buffer('pe', pe)

  def forward(self, x):
      x = x + self.pe[:,:x.shape[1],:].requires_grad_(False)
      return self.dropout(x)

class Generator(nn.Module):
  def __init__(self, n_embed, output_vocab_size):
      super().__init__()
      self.proj = nn.Linear(n_embed, output_vocab_size)

  def forward(self, x):
      return self.proj(x)

class EncoderDecoderTransformer(nn.Module):
  def __init__(self, n_heads, n_embed, dropout, n_blocks, ff_expansion_factor, context_size, input_vocab_size, output_vocab_size, pad_idx, device, store_attention=False):
      super().__init__()
      self.src_embedding = nn.Embedding(input_vocab_size, n_embed, padding_idx=pad_idx)
      self.targ_embedding = nn.Embedding(output_vocab_size, n_embed, padding_idx=pad_idx)
      self.src_positional_embedding = PositionalEmbedding(n_embed, context_size, device)
      self.targ_positional_embedding = PositionalEmbedding(n_embed, context_size, device)


      self.encoders = nn.Sequential(*[EncoderBlock(n_heads, n_embed, ff_expansion_factor, dropout, store_attention) for _ in range(n_blocks)])
      self.decoders = nn.Sequential(*[DecoderBlock(n_heads, n_embed, ff_expansion_factor, dropout, store_attention) for _ in range(n_blocks)])

      self.output = Generator(n_embed, output_vocab_size)

      self.scale = n_embed ** 0.5

  def encode(self, src, src_mask):
      x = self.src_embedding(src) * self.scale
      x = self.src_positional_embedding(x)
      x = self.encoders((x, src_mask))[0]
      return x

  def decode(self, targ, features, src_mask, targ_mask):
      targ = self.targ_embedding(targ) * self.scale
      targ = self.targ_positional_embedding(targ)
      out = self.decoders((targ , features, src_mask, targ_mask))[0]

      return self.output(out)

  def forward(self, src, targ, src_mask, targ_mask):
      x = self.encode(src, src_mask)
      out = self.decode(targ, x, src_mask, targ_mask)
      return out