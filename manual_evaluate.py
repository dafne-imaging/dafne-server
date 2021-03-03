import sys
import os
import re
import time

from dl.DynamicDLModel import DynamicDLModel
from utils import evaluate_model

model_path = os.path.abspath(sys.argv[1])
m = re.search(r'/models/([^/_]+)', model_path)
if not m:
    print('Invalid path to model - cannot determine the model type')
    sys.exit(1)

model_type = m.group(1)
print('Model type:', model_type)

print('Loading model...')
model = DynamicDLModel.Load(open(model_path, 'rb'))

print('Evaluating model...')
t = time.time()
dice = evaluate_model(model_type, model, save_log=False)
elapsed = time.time() - t
print('Dice score:', dice)
print('Elapsed time', elapsed)
