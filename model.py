import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size, embed=True):
    super(Encoder, self).__init__() 
    self.embed = embed
    if self.embed:
      self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=3)
    self.encoder = nn.LSTM(emb_size, hid_size)

  def pad(self, arr, pad=3):
    # Given an array of integer arrays, pad all arrays to the same length
    lengths = [len(e) for e in arr]
    max_len = max(lengths)
    return [e+[pad]*(max_len-len(e)) for e in arr], lengths

  def forward(self, seqs, lens):
    if type(seqs[0][0]) is int:
      seqs, lens = self.pad(seqs) 
      seqs = torch.cuda.LongTensor(seqs).t()

    # Embed
    if self.embed:
      emb_seqs = self.embedding(seqs)
    else:
      emb_seqs = seqs

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

class HierarchicalEncoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size):
    super(HierarchicalEncoder, self).__init__() 
    self.utt_encoder = Encoder(vocab_size, emb_size, hid_size, embed=True)
    self.ctx_encoder = Encoder(vocab_size, hid_size, hid_size, embed=False)

  def forward(self, seqs, lens):
    # First split into utterances
    utts = []
    conv_inds = []
    for i,seq in enumerate(seqs):
      cur_seq = seq
      while len(cur_seq) > 0:
        # Find and add full utt
        next_utt_ind = cur_seq.index(1)+1
        utts.append(cur_seq[:next_utt_ind])
        cur_seq = cur_seq[next_utt_ind:]
        conv_inds.append(i)

    # Encode all of the utterances
    _, encoder_hiddens = self.utt_encoder(utts, None)

    # Re-construct conversations
    ctx_hiddens = [[] for _ in range(len(lens))]
    for i,ind in enumerate(conv_inds):
      ctx_hiddens[ind].append(encoder_hiddens[0][0,i])

    # Pad hidden states and create a tensor
    ctx_lens = [len(ctx) for ctx in ctx_hiddens]
    max_ctx_len = max(ctx_lens)
    hid_size = ctx_hiddens[0][0].size()
    ctx_hiddens = [ctx+[torch.zeros(hid_size).cuda()]*(max_ctx_len-len(ctx))
                   for ctx in ctx_hiddens]
    ctx_tensor = torch.stack([torch.stack(ctx) for ctx in ctx_hiddens]).permute(1,0,2)

    return self.ctx_encoder(ctx_tensor, ctx_lens)

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
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    if use_attn:
      self.decoder = nn.LSTM(emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      output, hidden = self.decoder(embedded, hidden)
      return F.log_softmax(self.out(output), dim=2), hidden
    else:
      embedded = self.embedding(last_word)

      # Attn
      h = hidden[0].repeat(encoder_outputs.size(0), 1, 1)
      attn_energy = F.tanh(self.W_a(torch.cat((h, encoder_outputs), dim=2)))
      attn_logits = self.v(attn_energy).squeeze(-1) - 1e5 * (encoder_outputs.sum(dim=2) == 0).float()
      attn_weights = F.softmax(attn_logits, dim=0).permute(1,0).unsqueeze(1)
      context_vec = attn_weights.bmm(encoder_outputs.permute(1,0,2)).permute(1,0,2)

      # Concat with embeddings
      rnn_input = torch.cat((context_vec, embedded), dim=2)

      # Forward
      output, hidden = self.decoder(rnn_input, hidden)
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

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    decoder_hidden = self.policy(encoder_hidden, db, bs)

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)

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

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = input_seq.size(1)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(input_seq.size(1))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(input_seq.size(1))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert input_seq.size(1) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(input_seq.size(1))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(input_seq.size(1))]])
      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)

          # Get top candidates and add new hypotheses
          topv, topi = decoder_output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
