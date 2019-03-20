import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size):
    super(Encoder, self).__init__() 
    self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=3)
    self.encoder = nn.LSTM(emb_size, hid_size)

  def forward(self, seqs, lens):
    # Embed
    emb_seqs = self.embedding(seqs)

    # Sort by length
    sort_idx = sorted(range(len(lens)), key=lambda i: -lens[i])
    emb_seqs = emb_seqs[:,sort_idx]
    lens = [lens[i] for i in sort_idx]

    # Pack sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(emb_seqs, lens)

    # Forward pass through LSTM
    outputs, hidden = self.encoder(packed)

    # Unpack outputs
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    
    # Unsort
    unsort_idx = sorted(range(len(lens)), key=lambda i: sort_idx[i])
    outputs = outputs[:,unsort_idx]
    hidden = (hidden[0][:,unsort_idx], hidden[1][:,unsort_idx])

    return outputs, hidden

class Policy(nn.Module):
  def __init__(self, hidden_size, db_size, bs_size):
    super(Policy, self).__init__()
    self.proj_hid = nn.Linear(hidden_size, hidden_size)
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_bs = nn.Linear(bs_size, hidden_size)

  def forward(self, hidden, db, bs):
    output = self.proj_hid(hidden[0]) + self.proj_db(db) + self.proj_bs(bs)
    return (F.tanh(output), hidden[1])

class Decoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, use_attn=True):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.decoder = nn.LSTM(emb_size, hid_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    if use_attn:
      print("ERROR: NO ATTN")

  def forward(self, hidden, last_word):
    embedded = self.embedding(last_word)
    output, hidden = self.decoder(embedded, hidden)
    return F.log_softmax(self.out(output), dim=2), hidden


class Model(nn.Module):
  def __init__(self, encoder, policy, decoder, input_w2i, output_w2i, args):
    super(Model, self).__init__()

    self.args = args

    # Model
    self.encoder = encoder
    self.policy = policy
    self.decoder = decoder

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    input_seq = torch.cuda.LongTensor(input_seq).t()

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    target_seq = torch.cuda.LongTensor(target_seq).t()

    db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
    bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    decoder_hidden = self.policy(encoder_hidden, db, bs)

    # Decoder
    last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(target_seq.size(1))]])
    for t in range(target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word)

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, target_seq, target_lens, db, bs):

    batch_size = target_seq.size(1)
    predictions = torch.zeros((batch_size, target_seq.size(0)))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(target_seq.size(1))]])
      for t in range(target_seq.size(0)):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word)

        # Get top candidates
        topv, topi = decoder_out.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(-1, 1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
          word = self.output_index2word(str(int(ind.item())))
          if word == '_EOS':
              break
          sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences


  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
