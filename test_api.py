import requests
import dill

from dl.DynamicDLModel import DynamicDLModel

"""
Place model weights for example here: `models/thigh/1603281013.model`.  
Start server by running `python serve.py`.  
Then in another shell run `python test_api.py`.  
"""

url_base = 'http://localhost:5000/'


print("------------- info model ------------------")

r = requests.post(url_base + "info_model",
                  json={"model_type": "thigh", "api_key": "abc123"})
if r.ok:
    latest_timestamp = r.json()['latest_timestamp']
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")



print("------------- Get model ------------------")

r = requests.post(url_base + "get_model",
                  json={"model_type": "thigh", "timestamp": latest_timestamp, "api_key": "abc123"})
if r.ok:
    model = DynamicDLModel.Loads(r.content)
    model.dump(open('new_model.model', 'wb'))
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")



print("------------- Upload model ------------------")
model = DynamicDLModel.Load(open('new_model.model', 'rb'))
files = {'model_binary': model.dumps()}
r = requests.post(url_base + "upload_model", files=files,
                  data={"model_type": "thigh", "api_key": "abc123"})
print(f"status code: {r.status_code}")
print(f"message: {r.json()['message']}")
