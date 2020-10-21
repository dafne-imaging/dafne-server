import requests

"""
Start server by running `python serve.py`.
Then you can access the API like the following
"""

url_base = 'http://localhost:5000/'


print("------------- Get model ------------------")

r = requests.post(url_base + "get_model",
                  json={"name": "thigh_segmentation_test", "access_code": "123"})
if r.ok:
    with open('new_weights.hdf5', 'wb') as f:
        f.write(r.content)
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


print("------------- Upload model ------------------")

files = {'upload_file': open('/Users/jakob/dev/dafne-server/new_weights.hdf5','rb')}
r = requests.post(url_base + "upload_model", files=files,
                  data={"name": "thigh_segmentation_test", "access_code": "123"})
print(f"status code: {r.status_code}")
print(f"message: {r.json()['message']}")
