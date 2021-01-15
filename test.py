import os

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set to 2 to hide all warnings

import requests
import dill

from dl.DynamicDLModel import DynamicDLModel
from utils import MODELS_DIR, evaluate_model

model_path = f"{MODELS_DIR}/Thigh/1603281020.model"
model = DynamicDLModel.Load(open(model_path, "rb"))

score = evaluate_model("Leg", model)

print(f"score: {score}")
