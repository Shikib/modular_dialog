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
      #if self.args.use_cuda is True: 
      #  seqs = torch.cuda.LongTensor(seqs).t()
      #else:
      #  seqs = torch.LongTensor(seqs).t()

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

class PNN(nn.Module):
  def __init__(self, hidden_size, db_size):
    super(PNN, self).__init__()
    self.proj_hid = nn.Linear(hidden_size, hidden_size)
    self.proj_db = nn.Linear(db_size, hidden_size)

  def forward(self, hidden, db):
    output = self.proj_hid(hidden) + self.proj_db(db)
    return F.tanh(output)

class Decoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, use_attn=True):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    self.hid_size = hid_size
    if use_attn:
      self.decoder = nn.LSTM(emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs, ret_out=False):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      output, hidden = self.decoder(embedded, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output
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
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output

class ColdFusionDecoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, use_attn=True):
    super(ColdFusionDecoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    self.hid_size = hid_size
    if use_attn:
      self.decoder = nn.LSTM(vocab_size+emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(vocab_size+emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs, lm_preds, ret_out=False):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      rnn_input = torch.cat((lm_preds, embedded), dim=2)
      output, hidden = self.decoder(rnn_input, hidden)
      if not return_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output
    else:
      embedded = self.embedding(last_word)

      # Attn
      h = hidden[0].repeat(encoder_outputs.size(0), 1, 1)
      attn_energy = F.tanh(self.W_a(torch.cat((h, encoder_outputs), dim=2)))
      attn_logits = self.v(attn_energy).squeeze(-1) - 1e5 * (encoder_outputs.sum(dim=2) == 0).float()
      attn_weights = F.softmax(attn_logits, dim=0).permute(1,0).unsqueeze(1)
      context_vec = attn_weights.bmm(encoder_outputs.permute(1,0,2)).permute(1,0,2)

      # Concat with embeddings
      rnn_input = torch.cat((context_vec, embedded, lm_preds.unsqueeze(0)), dim=2)

      # Forward
      output, hidden = self.decoder(rnn_input, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output

class ColdFusionLayer(nn.Module):
  def __init__(self, hid_size, vocab_size):
    super(ColdFusionLayer, self).__init__()
    self.lm_proj = nn.Linear(vocab_size, hid_size)
    self.mask_proj = nn.Linear(2*hid_size, hid_size)
    self.out = nn.Linear(2*hid_size, vocab_size)

  def forward(self, dec_hid, lm_pred):
    lm_hid = self.lm_proj(lm_pred)
    mask = F.sigmoid(self.mask_proj(torch.cat((dec_hid, lm_hid), dim=2)))
    return F.log_softmax(self.out(torch.cat((dec_hid, lm_hid*mask), dim=2)), dim=2)

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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

class LanguageModel(nn.Module):
  def __init__(self, decoder, input_w2i, output_w2i, args):
    super(LanguageModel, self).__init__()

    self.args = args
    assert not decoder.use_attn

    # Model
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
    decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                      torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                        torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

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
      assert len(input_seq) == 1

      decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                        torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), None)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], None)

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
    torch.save(self.decoder, name+'.dec')

  def load(self, name):
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())

class ShallowFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, args):
    super(ShallowFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
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

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)
      lm_output, lm_hidden = self.lm(lm_hidden, last_word, None)

      # Save output
      probas[t] = decoder_output + lm_output

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)
        lm_output, lm_hidden = self.lm(lm_hidden, last_word, None)

        # Get top candidates
        topv, topi = (decoder_output + lm_output).data.topk(1)
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

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs)
            lm_output, lm_hidden = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[2], encoder_outputs)
            lm_output, lm_hidden = self.lm(hyp[1], hyp[2], None)

          # Get top candidates and add new hypotheses
          topv, topi = (decoder_output+lm_output).data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
                       _to_cpu(last_word), 
                       hyp[3] + topv[i], 
                       hyp[4] + 1, 
                       hyp[5] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      finished = [hyp for hyp in beam if hyp[-1].endswith('_EOS')]
      if len(finished) == 0:
        finished = beam
      return [max(finished, key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')
    torch.save(self.lm, name+'.lmdec')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())

class DeepFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, args):
    super(DeepFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder
    self.out = nn.Linear(self.decoder.hid_size + self.lm.hid_size, len(seq2seq.output_w2i))

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
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

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
      _, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

      # Save output
      probas[t] = F.log_softmax(self.out(F.dropout(torch.cat((decoder_output.detach(), lm_output.detach()), dim=2), p=0.5)), dim=2)

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
        _, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

        # Save output
        output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

        # Get top candidates
        topv, topi = output.data.topk(1)
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

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            _, decoder_hidden, decoder_output = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)
          else:
            _, decoder_hidden, decoder_output = self.decoder(hyp[0], hyp[2], encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(hyp[1], hyp[2], None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

          # Get top candidates and add new hypotheses
          topv, topi = output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
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
    torch.save(self.lm, name+'.lmdec')
    torch.save(self.out, name+'.out')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())
    self.out.load_state_dict(torch.load(name+'.out').state_dict())

class ColdFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, cf, args):
    super(ColdFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder
    self.cf = cf

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
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

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
      lm_pred, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

      # Save output
      probas[t] = self.cf(decoder_output, lm_pred.detach())

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
        lm_pred, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

        # Save output
        output = self.cf(decoder_output, lm_pred.detach())

        # Get top candidates
        topv, topi = output.data.topk(1)
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

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            _, decoder_hidden, decoder_output = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)
          else:
            _, decoder_hidden, decoder_output = self.decoder(hyp[0], hyp[2], encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(hyp[1], hyp[2], None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

          # Get top candidates and add new hypotheses
          topv, topi = output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
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
    torch.save(self.lm, name+'.lmdec')
    torch.save(self.cf, name+'.cf')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())
    self.cf.load_state_dict(torch.load(name+'.cf').state_dict())



class DialogActToResponseGeneration(nn.Module):

  def __init__(self, decoder, hid_size, da_size, db_size, output_w2i, args):

    super(DialogActToResponseGeneration, self).__init__()

    self.args = args

    self.proj_da = nn.Linear(da_size, hid_size)   # dialog act projection
    self.proj_db = nn.Linear(db_size, hid_size)   # Database projection
    self.decoder = decoder

    # Vocab
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

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])

    return target_seq, target_lens, db, da


  def forward(self, target_seq, target_lens, db, da):

    db_proj = self.db_proj(db)
    da_proj = self.da_proj(da)

    # Policy network
    decoder_hidden = F.tanh(da_proj + db_proj)

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


  def train(self, target_seq, target_lens, db, da):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(target_seq, target_lens, db, da)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, max_len, db, da):
    batch_size = len(db)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      db_proj = self.db_proj(db)
      da_proj = self.da_proj(da)

      # Policy network
      decoder_hidden = F.tanh(da_proj + db_proj)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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

  def beam_decode(self, max_len, db, da, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

  def save(self, name):
    torch.save(self, name+'.da2resp')


  def load(self, name):
    self.load_state_dict(torch.load(name+'.da2resp').state_dict())





#class ColdFusionModel(nn.Module):
#  def __init__(self, encoder, policy, decoder, lm, input_w2i, output_w2i, args):
#    super(ColdFusionModel, self).__init__()
#
#    self.args = args
#
#    # Model
#    self.encoder = encoder
#    self.policy = policy
#    self.decoder = decoder
#    self.lm = lm
#
#    # Vocab
#    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
#    self.input_w2i = input_w2i
#    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
#    self.output_w2i = output_w2i
#
#    # Training
#    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
#    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
#
#  def prep_batch(self, rows, hierarchical=True):
#    def _pad(arr, pad=3):
#      # Given an array of integer arrays, pad all arrays to the same length
#      lengths = [len(e) for e in arr]
#      max_len = max(lengths)
#      return [e+[pad]*(max_len-len(e)) for e in arr], lengths
#
#    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
#    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
#    # if self.args.use_cuda is True:
#    #   input_seq = torch.cuda.LongTensor(input_seq).t()
#    # else:
#    #   input_seq = torch.LongTensor(input_seq).t()
#    input_seq = inputs
#    input_lens = [len(inp) for inp in input_seq]
#
#    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
#    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
#    if self.args.use_cuda is True:
#      target_seq = torch.cuda.LongTensor(target_seq).t()
#    else:
#      target_seq = torch.LongTensor(target_seq).t()
#
#    if self.args.use_cuda is True:
#      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
#      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
#    else:
#      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
#      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
#
#    return input_seq, input_lens, target_seq, target_lens, db, bs
#
#  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
#    # Pass through LM
#    lm_probas = self.lm(input_seq, input_lens, target_seq, target_lens, db, bs).detach()
#
#    # Encoder
#    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#    # Policy network
#    decoder_hidden = self.policy(encoder_hidden, db, bs)
#
#    # Decoder
#    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
#    if self.args.use_cuda is True:
#      probas = probas.cuda()
#    last_word = target_seq[0].unsqueeze(0)
#    for t in range(1,target_seq.size(0)):
#      # Pass through decoder
#      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, lm_probas[t])
#
#      # Save output
#      probas[t] = decoder_output
#
#      # Set new last word
#      last_word = target_seq[t].unsqueeze(0)
#
#    return lm_probas
#
#  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
#    self.optim.zero_grad()
#
#    # Forward
#    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)
#
#    # Loss
#    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
#
#    # Backwards
#    loss.backward()
#    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
#    self.optim.step()
#
#    return loss.item()
#
#  def decode(self, input_seq, input_lens, max_len, db, bs):
#
#    batch_size = len(input_seq)
#    predictions = torch.zeros((batch_size, max_len))
#
#    with torch.no_grad():
#      # Encoder
#      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#      # Policy network
#      decoder_hidden = self.policy(encoder_hidden, db, bs)
#      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
#                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())
#
#      # Decoder
#      if self.args.use_cuda is True:
#        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      else:
#        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      for t in range(max_len):
#        lm_probas, lm_hidden = self.lm.decoder(lm_hidden, last_word, None)
#
#        # Pass through decoder
#        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, lm_probas.squeeze(0))
#
#        # Get top candidates
#        topv, topi = decoder_output.data.topk(1)
#        topi = topi.view(-1)
#
#        predictions[:, t] = topi
#
#        # Set new last word
#        last_word = topi.detach().view(1, -1)
#
#    predicted_sentences = []
#    for sentence in predictions:
#      sent = []
#      for ind in sentence:
#        word = self.output_i2w[ind.long().item()]
#        if word == '_EOS':
#          break
#        sent.append(word)
#      predicted_sentences.append(' '.join(sent))
#
#    return predicted_sentences
#
#  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
#    assert False, "not implemented yet"
#    def _to_cpu(x):
#      if type(x) in [tuple, list]:
#        return [e.cpu() for e in x]
#      else:
#        return x.cpu()
#
#    def _to_cuda(x):
#      if type(x) in [tuple, list]:
#        return [e.cuda() for e in x]
#      else:
#        return x.cuda()
#
#    def _score(hyp):
#      return hyp[2]/float(hyp[3] + 1e-6) 
#
#    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
#    with torch.no_grad():
#      # Batch size must be 1
#      assert len(input_seq) == 1
#
#      # Encoder
#      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#      # Policy network
#      decoder_hidden = self.policy(encoder_hidden, db, bs)
#
#      # Decoder
#      if self.args.use_cuda is True:
#        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      else:
#        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
#      for _ in range(max_len):
#        new_beam = []
#        for hyp in beam:
#          # Continue _EOS
#          if hyp[-1].endswith('_EOS'):
#            new_beam.append(hyp)
#            continue
#          
#          # Propagate through decoder
#          if self.args.use_cuda is True:
#            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs)
#          else:
#            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)
#
#          # Get top candidates and add new hypotheses
#          topv, topi = decoder_output.data.topk(beam_width)
#          topv = topv.squeeze()
#          topi = topi.squeeze()
#          for i in range(beam_width):
#            last_word = topi[i].detach().view(1, -1)
#            new_hyp = (_to_cpu(decoder_hidden), 
#                       _to_cpu(last_word), 
#                       hyp[2] + topv[i], 
#                       hyp[3] + 1, 
#                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])
#
#            new_beam.append(new_hyp)
#
#        # Translate new beam into beam
#        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]
#
#      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
#          
#  def save(self, name):
#    torch.save(self.encoder, name+'.enc')
#    torch.save(self.policy, name+'.pol')
#    torch.save(self.decoder, name+'.dec')
#    torch.save(self.lm, name+'.lm')
#
#  def load(self, name):
#    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
#    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
#    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
#    self.lm.load_state_dict(torch.load(name+'.lm').state_dict())

class NLU(nn.Module):
  def __init__(self, encoder, input_w2i, args):
    super(NLU, self).__init__() 

    # Model
    self.encoder = encoder
    self.linear = nn.Linear(args.hid_size, args.bs_size)

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i

    # Training
    self.criterion = nn.BCEWithLogitsLoss(reduce=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args


  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    input_seq = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    input_lens = [len(inp) for inp in input_seq]

    if self.args.use_cuda is True:
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, bs


  def forward(self, input_seq, input_lens):
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
    bs_out_logits = self.linear(encoder_hidden[0])

    return bs_out_logits.squeeze(0)


  def train(self, input_seq, input_lens, bs):
    self.optim.zero_grad()

    # Forward
    bs_out_logits = self.forward(input_seq, input_lens)

    # Loss
    loss = self.criterion(bs_out_logits, bs)

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def predict(self, input_seq, input_lens):
    out_logits = self.forward(input_seq, input_lens)
    out_probs = F.sigmoid(out_logits)
    return out_probs >= 0.5

  def save(self, name):
    torch.save(self, name+'.nlu')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.nlu').state_dict())

class DM(nn.Module):
  def __init__(self, pnn, args):
    super(DM, self).__init__() 

    # Model
    self.bs_in = nn.Linear(args.bs_size, args.hid_size)
    self.pnn = pnn
    self.da_out = nn.Linear(args.hid_size, args.da_size)

    # Training
    self.criterion = nn.BCEWithLogitsLoss(reduce=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    if self.args.use_cuda is True:
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
    else:
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])

    return bs, da, db

  def forward(self, bs, db):
    bs_enc = self.bs_in(bs)
    da_hid = self.pnn(bs_enc, db)
    da_pred = self.da_out(da_hid)
    return da_pred

  def train(self, bs, da, db):
    self.optim.zero_grad()

    # Forward
    logits = self.forward(bs, db)

    # Loss
    loss = self.criterion(logits, da)

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def predict(self, bs, db):
    out_logits = self.forward(bs, db)
    out_probs = F.sigmoid(out_logits)
    return out_probs >= 0.5

  def save(self, name):
    torch.save(self, name+'.dm')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.dm').state_dict())

class NLG(nn.Module):
  def __init__(self, decoder, output_w2i, args):
    super(NLG, self).__init__()

    # Model
    self.da_in = nn.Linear(args.da_size, args.hid_size)
    self.db_in = nn.Linear(args.db_size, args.hid_size)
    self.decoder = decoder

    # Vocab
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=output_w2i['_PAD'], size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])

    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])

    return target_seq, target_lens, db, da

  def forward(self, target_seq, target_lens, db, da):
    db_proj = self.db_in(db).unsqueeze(0)
    da_proj = self.da_in(da).unsqueeze(0)

    # Policy network
    decoder_hidden = (F.tanh(da_proj + db_proj), F.tanh(da_proj + db_proj))

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, target_seq, target_lens, db, da):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(target_seq, target_lens, db, da)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, max_len, db, da):
    batch_size = len(db)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      db_proj = self.db_in(db)
      da_proj = self.da_in(da)

      # Policy network
      decoder_hidden = (F.tanh(da_proj + db_proj).unsqueeze(0), F.tanh(da_proj + db_proj).unsqueeze(0))

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(batch_size)]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(batch_size)]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

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

  def save(self, name):
    torch.save(self, name+'.nlg')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.nlg').state_dict())

class E2E(nn.Module):
  def __init__(self, encoder, pnn, decoder, input_w2i, output_w2i, args):
    super(E2E, self).__init__()

    # Model
    self.encoder = encoder
    self.pnn = pnn
    self.decoder = decoder

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    input_seq = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
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
    decoder_hidden = (self.pnn(encoder_hidden[0], db), self.pnn(encoder_hidden[0], db))

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
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = (self.pnn(encoder_hidden[0], db), self.pnn(encoder_hidden[0], db))

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = (self.pnn(encoder_hidden[0], db), self.pnn(encoder_hidden[0], db))

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
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
    torch.save(self, name+'.e2e')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.e2e').state_dict())
