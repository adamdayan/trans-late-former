import torch
import datetime
import random
import numpy as np
from tqdm import tqdm
import en_core_web_sm
import de_core_news_sm
import torchtext.datasets

import translateformer as tlf

# detect device
device= 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device detected as {device}")

# set seeds 
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# get datasets
print("getting datasets")
train_data = torchtext.datasets.Multi30k(split='train', language_pair=('de', 'en'))
val_data = torchtext.datasets.Multi30k(split='valid', language_pair=('de', 'en'))
test_data = torchtext.datasets.Multi30k(split='test', language_pair=('de', 'en'))

# load tokenizers
en_tokenizer = en_core_web_sm.load()
de_tokenizer = de_core_news_sm.load()


# build vocab
print("pre-processing data")
src_text, targ_text = list(zip(*train_data))

min_freq = 2

de_toks = tlf.get_unique_tokens(src_text, de_tokenizer, min_freq)
en_toks = tlf.get_unique_tokens(targ_text, en_tokenizer, min_freq)
print(f"{len(de_toks)=} {len(en_toks)=}")

de_vocab = tlf.Vocab(de_tokenizer, de_toks)
en_vocab = tlf.Vocab(en_tokenizer, en_toks)

# create dataloaders from data set
train_dataloader = tlf.make_dataloader(train_data, 128, 100, de_vocab, en_vocab, device)
val_dataloader = tlf.make_dataloader(val_data, 128, 100, de_vocab, en_vocab, device)

# setup model
print("setting up model")
model = tlf.EncoderDecoderTransformer(
    n_heads=8, n_embed=256, dropout=0.1, n_blocks=3, ff_expansion_factor=2, 
    context_size=100, input_vocab_size=len(de_vocab.tokens), output_vocab_size=len(en_vocab.tokens), 
    pad_idx=en_vocab.PAD_IDX, device=device, store_attention=True
).to(device)

model = model.apply(tlf.initialize_weights)

criterion = torch.nn.CrossEntropyLoss(ignore_index = en_vocab.PAD_IDX)

optim = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)

# train loop

train_losses = []
val_losses = []
cur_best_loss = float("inf")

n_epochs = 7

print("training for {n_epochs} epochs")
for epoch in range(n_epochs):
    print("beginning epoch {epoch}")
    start_time = datetime.datetime.utcnow()

    train_loss = tlf.train(model, train_dataloader, criterion, optim)
    val_loss = tlf.evaluate(model, val_dataloader, criterion)

    end_time = datetime.datetime.utcnow()
    print(f"{train_loss=} {val_loss=} duration={end_time - start_time}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < cur_best_loss:
      cur_best_loss = val_loss
      torch.save(model.state_dict(), 'output/trans-late-former-model.pt')


# load best performing model
model.load_state_dict(torch.load('output/trans-late-former-model.pt'))

# calculate bleu
bleu = tlf.calculate_bleu(model, val_dataloader, 100, de_vocab.SOS_IDX, en_vocab.PAD_IDX, de_vocab.EOS_IDX, en_vocab)
print(f"BLEU score: {bleu}")

# generate attention figures 
src = next(iter(train_dataloader))[0][0]
pred_targ = tlf.greedy_decoding(model, src.unsqueeze(0), src.shape[0], en_vocab.SOS_IDX, en_vocab.PAD_IDX)
attn = model.decoders[0].cross_attention.attn_store

first_pad_idx = torch.argmax((src == de_vocab.PAD_IDX).to(dtype=torch.int), dim=-1).item()
trunc_src = src[:first_pad_idx]
pred_trans = [""] + [en_vocab.textify(tok) for tok in pred_targ[0]] + ["<eos>"]
src_trans =  [""] + [de_vocab.textify(tok) for tok in trunc_src]
attn = attn[:, :, :, :first_pad_idx]


tlf.visualise_attention(attn, src_trans, pred_trans, save_path="output/attention_decoder_0.png")