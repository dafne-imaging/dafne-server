import os
import requests
import dill

from dl.DynamicDLModel import DynamicDLModel

"""
Place model weights for example here: `db/models/Thigh/1603281013.model`.  
Start server by running `python main.py`.  
Then in another shell run `python test_api.py`.  
"""

# url_base = 'http://localhost:5000/'
url_base = 'http://www.dafne.network:5000/'


print("------------- info model ------------------")

r = requests.post(url_base + "info_model",
                  json={"model_type": "Thigh", "api_key": "abc123"})
if r.ok:
    latest_timestamp = r.json()['latest_timestamp']
    print(f"latest_timestamp: {latest_timestamp}")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")



print("------------- Get model ------------------")

r = requests.post(url_base + "get_model",
                  json={"model_type": "Thigh", "timestamp": latest_timestamp, "api_key": "abc123"})
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
                  data={"model_type": "Thigh", "api_key": "abc123"})
print(f"status code: {r.status_code}")
print(f"message: {r.json()['message']}")



os.remove("new_model.model")  # Delete temporary file


# print("------------- my_test ------------------")

# r = requests.post(url_base + "my_test",
#                   json={"model_type": "Thigh", "api_key": "abc123"})
# if r.ok:
#     latest_timestamp = r.json()['latest_timestamp']
#     print(f"latest_timestamp: {latest_timestamp}")
# else:
#     print(f"status code: {r.status_code}")
#     print(f"message: {r.json()['message']}")


