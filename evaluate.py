import argparse
import json
import math
import random

from utils.nlp import *

predict_f = open('baseline_19_predicted_sentences.txt')
predicted_sentences = [[x.strip()] for x in predict_f][:-1]

target_f = open('attn_19_target_sentences.txt')
target_sentences = [[x.strip()] for x in target_f][:-1]

bscorer = BLEUScorer()

bleu_score = bscorer.score(predicted_sentences, target_sentences)

print("Test BLEU score: {0:.10f}".format(bleu_score))
