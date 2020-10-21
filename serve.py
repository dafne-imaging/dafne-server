import os, glob, time

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from markupsafe import escape

from utils import valid_credentials, get_model_types, get_models, get_username, merge_model, log

app = Flask(__name__)


@app.route('/info_model', methods=["POST"])
def info_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    latest_timestamp = get_models(meta["model_type"])[-1]
    return {"latest_timestamp": latest_timestamp}, 200


@app.route('/get_model', methods=["POST"])
def get_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401
    model = "models/" + meta["model_type"] + "/" + \
            meta["timestamp"] + ".hdf5"  # todo: read and write access only to models dir
    if not os.path.isfile(model):
        return {"message": "invalid model - not found"}, 500

    username = get_username(meta["api_key"])
    log(f"get_model accessed by {username} - {meta['model_type']} - {meta['timestamp']}")
    return send_file(model), 200


@app.route('/upload_model', methods=['POST'])
def upload_file():
    meta = request.form.to_dict()
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401
    if meta["model_type"] not in get_model_types():
        return {"message": "invalid model type"}, 500
            
    f = request.files['upload_file']
    username = get_username(meta["api_key"])
    model_path = "models/" + meta["model_type"] + "/uploads/" + str(int(time.time())) + \
                 "_" + username + ".hdf5"
    f.save(model_path)
    merge_model(meta["model_type"], model_path)
    log(f"upload_model accessed by {username} - {meta['model_type']} - {model_path}")
    return {"message": "upload successful"}, 200
            

if __name__ == '__main__':
    app.run(debug=True)
