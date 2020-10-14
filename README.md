# Dafne-Server

This is the backend server for the Dafne platform.

The server does the following:
1. provide trained models to the client
2. receive updated models from the client
3. merge the new client models with the existing model to improve the model (federated learning)
4. provide new improved model to the client

See [test_api.py](test_api.py) for how to use the API from the client.