# Dafne-Server

This is the backend server for the Dafne platform.

The server does the following:
1. provide trained models to the client
2. receive updated models from the client
3. merge the new client models with the existing model to improve the model (federated learning)
4. provide new improved model to the client

## How to start the server
Put models into the following folder structure:
```
model
'-> Classifier
    '-> XXX.model
'-> Leg
    '-> XXX.model
'-> Thigh
    '-> XXX.model
```
Replace XXX with an integer (unique ID / timestamp) (e.g. 1603281013.model).  
```
python serve.py
```


### How to use the API
See [test_api.py](test_api.py)