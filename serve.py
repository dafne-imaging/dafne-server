import glob

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from markupsafe import escape

from utils import valid_credentials

app = Flask(__name__)

def get_available_models():
    available_models = [f for f in glob.glob("models/*.hdf5")]
    return [m.split("/")[1].split(".")[0] for m in available_models]

@app.route('/info_model', methods=["POST"])
def info_model():
    if valid_credentials(request.json["username"], request.json["pwd"]):
        model = "models/weights_coscia.hdf5"
        return send_file(model), 200
    else:
        return {"message": "invalid access code"}, 401


@app.route('/get_model', methods=["POST"])
def get_model():
    meta = request.json
    if is_valid_access_code(meta["access_code"]):
        model = meta["name"]
        if model in get_available_models():
            model = "models/" + model + ".hdf5"  # todo: read and write access only to models dir
            return send_file(model), 200
        else:
            return {"message": "invalid model name - not found"}, 500
    else:
        return {"message": "invalid access code"}, 401


@app.route('/upload_model', methods=['POST'])
def upload_file():
    meta = request.form.to_dict()
    if is_valid_access_code(meta["access_code"]):
        model = meta["name"]
        if model in get_available_models():
            f = request.files['upload_file']
            f.save("models/" + model + ".hdf5")
            return {"message": "upload successful"}, 200
        else:
            return {"message": "invalid model name - not found"}, 500
    else:
        return {"message": "invalid access code"}, 401


if __name__ == '__main__':
    app.run(debug=True)
