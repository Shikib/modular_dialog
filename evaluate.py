import argparse
import json
import math
import random

from evaluate_model import evaluateModel

predictions = json.load(open('baseline_19_predictions.json'))
targets = json.load(open('data/test_dials.json'))

evaluateModel(predictions, targets, mode='test')
