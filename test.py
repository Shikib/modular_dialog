import argparse
import json
import math
import model
import random

from collections import defaultdict

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='MultiWoz Training Script')

parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--use_attn', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--model_name', type=str, default='baseline')

parser.add_argument('--emb_size', type=int, default=50)
parser.add_argument('--hid_size', type=int, default=150)
parser.add_argument('--db_size', type=int, default=30)
parser.add_argument('--bs_size', type=int, default=94)

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

args = parser.parse_args()

def load_data(filename):
  data = json.load(open(filename))
  rows = []
  for filename,dial in data.items():
    input_so_far = []
    for i in range(len(dial['sys'])):
      input_so_far += ['_GO'] + dial['usr'][i].strip().split() + ['_EOS']

      input_seq = [e for e in input_so_far]
      target_seq = ['_GO'] + dial['sys'][i].strip().split() + ['_EOS']
      db = dial['db'][i]
      bs = dial['bs'][i]

      rows.append((input_seq, target_seq, db, bs, filename, i))

      # Add sys output
      input_so_far += target_seq

  return rows

# Load vocabulary
input_w2i = json.load(open('data/input_lang.word2index.json'))
output_w2i = json.load(open('data/output_lang.word2index.json'))
input_i2w = json.load(open('data/input_lang.index2word.json'))
output_i2w = json.load(open('data/output_lang.index2word.json'))

# Create models
encoder = model.Encoder(vocab_size=len(input_w2i), 
                        emb_size=args.emb_size, 
                        hid_size=args.hid_size)

policy = model.Policy(hidden_size=args.hid_size,
                      db_size=args.db_size,
                      bs_size=args.bs_size)

decoder = model.Decoder(emb_size=args.emb_size,
                        hid_size=args.hid_size,
                        vocab_size=len(output_w2i),
                        use_attn=args.use_attn)

model = model.Model(encoder=encoder,
                    policy=policy,
                    decoder=decoder,
                    input_w2i=input_w2i,
                    output_w2i=output_w2i,
                    args=args).cuda()

# Load data
train = load_data('data/train_dials.json')
valid = load_data('data/val_dials.json')
test = load_data('data/test_dials.json')

val_targets = json.load(open('data/val_dials.json'))
test_targets = json.load(open('data/test_dials.json'))

num_val_batches = math.ceil(len(valid)/args.batch_size)
num_test_batches = math.ceil(len(test)/args.batch_size)

indices = list(range(len(test)))

model_name = 'attn'

best_val_score = 0.0
best_val_epoch = -1

for epoch in range(20):
  # Load saved model parameters
  model.load(model_name+"_"+str(epoch))
  all_predicted = defaultdict(list)
  for batch in range(num_val_batches):
    # Prepare batch
    batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
    batch_rows = [valid[i] for i in batch_indices]
    input_seq, input_lens, target_seq, target_lens, db, bs = model.prep_batch(batch_rows)

    # Get predicted sentences for batch
    predicted_sentences = model.decode(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Add predicted to list
    for i,sent in enumerate(predicted_sentences):
      all_predicted[batch_rows[i][-2]].append(sent) 

  val_score = evaluateModel(all_predicted, val_targets, mode='val')
  print("Epoch {0}: Validation Score {1:.10f}".format(epoch, val_score))
  print("-----------------------------------")

  if val_score > best_val_score:
    best_val_score = val_score
    best_val_epoch = epoch

print("Best validation score after epoch {0}".format(best_val_epoch))

# Evaluate best val model on test data
model.load(model_name+"_"+str(best_val_epoch))

all_predicted = defaultdict(list)
for batch in range(num_test_batches):
  # Prepare batch
  batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
  batch_rows = [test[i] for i in batch_indices]
  input_seq, input_lens, target_seq, target_lens, db, bs = model.prep_batch(batch_rows)

  # Get predicted sentences for batch
  predicted_sentences = model.decode(input_seq, input_lens, target_seq, target_lens, db, bs)

  # Add predicted to list
  for i,sent in enumerate(predicted_sentences):
    all_predicted[batch_rows[i][-2]].append(sent) 

test_score = evaluateModel(all_predicted, test_targets, mode='test')
print("Test score: {0}".format(test_score))