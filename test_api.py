import os
import requests
import dill

from dafne_dl.DynamicDLModel import DynamicDLModel
from dafne_dl.DynamicTorchModel import DynamicTorchModel
from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel

"""
Place model weights for example here: `db/models/Thigh/1603281013.model`.  
Start server by running `python main.py`.  
Then in another shell run `python test_api.py`.  
"""
url_base = 'http://localhost:5000/'
#url_base = 'https://www.dafne.network:5001/'

model_type = "CHP"
# model_type = "Thigh"
# model_type = "Leg"


print("------------- get available models ------------------")

r = requests.post(url_base + "get_available_models",
                  json={"api_key": "abc123"})
if r.ok:
    models = r.json()['models']
    print(f"models: {models}")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


print("------------- info model ------------------")

r = requests.post(url_base + "info_model",
                  json={"model_type": model_type, "api_key": "abc123"})
if r.ok:
    latest_timestamp = r.json()['latest_timestamp']
    print(f"latest_timestamp: {latest_timestamp}")
    print(r.json())
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


print("------------- Get model ------------------")

r = requests.post(url_base + "get_model",
                  json={"model_type": model_type, "timestamp": latest_timestamp, "api_key": "abc123"})
if r.ok:
    # model = DynamicDLModel.Loads(r.content)
    model = DynamicEnsembleModel.Loads(r.content)
    model.dump(open('new_model.model', 'wb'))
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


print("------------- Upload model ------------------")
model = DynamicEnsembleModel.Load(open('new_model.model', 'rb'))
# model = DynamicDLModel.Load(open('new_model.model', 'rb'))
files = {'model_binary': model.dumps()}
r = requests.post(url_base + "upload_model", files=files,
                  data={"model_type": model_type, "api_key": "abc123", "dice": 0.3})
print(f"status code: {r.status_code}")
print(f"message: {r.json()['message']}")

os.remove("new_model.model")  # Delete temporary file


print("------------- Upload data ------------------")
# filename = "db/test_data/Leg/leg_GRE_nsl2_21_20190204.npz"
filename = "db/test_data/Leg/CHP/subj_001.npz"
files = {'data_binary': open(filename, 'rb')}
r = requests.post(url_base + "upload_data", files=files,
                  data={"api_key": "abc123"})
print(f"status code: {r.status_code}")
print(f"message: {r.json()['message']}")


print("------------- Evaluate model ------------------")

r = requests.post(url_base + "evaluate_model",
                  json={"model_type": model_type, "timestamp": "1610001000", "api_key": "abc123"})
if r.ok:
    print(f"message: {r.json()['message']}")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


