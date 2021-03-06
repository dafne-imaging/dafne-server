import gc
import os, glob, time, io
from threading import Thread
import subprocess

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from markupsafe import escape

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set to 2 to hide all warnings

from dl.LocalModelProvider import LocalModelProvider
from dl.DynamicDLModel import DynamicDLModel
from utils import valid_credentials, get_model_types, get_models, get_username, merge_model, log
from utils import evaluate_model as utils_evaluate_model
from utils import MODELS_DIR
from dl.misc import calculate_file_hash

app = Flask(__name__)

# On the server /mnt/data/dafne-server-db will be mounted to db when starting the docker container
DB_DIR = "db"

@app.route('/get_available_models', methods=["POST"])
def get_available_models():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    return {"models": get_model_types()}, 200


@app.route('/info_model', methods=["POST"])
def info_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    latest_timestamp = get_models(meta["model_type"])[-1]
    latest_model_path = f"{MODELS_DIR}/{meta['model_type']}/{latest_timestamp}.model"
    model_hash = calculate_file_hash(latest_model_path, True)
    return {"latest_timestamp": latest_timestamp, "hash": model_hash}, 200


@app.route('/get_model', methods=["POST"])  
def get_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    # todo: read and write access only to models dir
    model = f"{MODELS_DIR}/{meta['model_type']}/{meta['timestamp']}.model"
    if not os.path.isfile(model):
        return {"message": "invalid model - not found"}, 500
    username = get_username(meta["api_key"])
    log(f"get_model accessed by {username} - {meta['model_type']} - {meta['timestamp']}")
    return send_file(model, mimetype='application/octet-stream'), 200


def merge_model_thread(model_type, new_model_path):
    subprocess.call(f"python standalone_merge.py {model_type} {new_model_path}", shell=True)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    available data fields:
        api_key
        model_type
        dice (optional)
    """
    meta = request.form.to_dict()
    if not valid_credentials(meta["api_key"]):
        log(f"Upload request of {meta['model_type']} rejected because api key {meta['api_key']} is invalid")
        return {"message": "invalid access code"}, 401

    username = get_username(meta["api_key"])

    if meta["model_type"] not in get_model_types():
        log(f"Upload request of {meta['model_type']} from {username} rejected because model type is invalid")
        return {"message": "invalid model type"}, 500

    #model_binary = request.files['model_binary'].read()  # read() is needed to get bytes from FileStorage object
    #model_delta = DynamicDLModel.Loads(model_binary)
    #model_orig_path = f"{MODELS_DIR}/{meta['model_type']}/{model_delta.timestamp_id}.model"
    #model_orig = DynamicDLModel.Load(open(model_orig_path, "rb")) # unused here - see merge_models
    # calc_delta(): new_model - orig_model = delta
    # apply_delta(): model_orig + delta = model_new
    # Model delta: only activate if rest of federated learning working properly
    # model = model_orig.apply_delta(model_delta)
    #model = model_delta

    # directly save received model to disk
    model_path = f"{MODELS_DIR}/{meta['model_type']}/uploads/{str(int(time.time()))}_{username}.model"
    request.files['model_binary'].save(model_path)

    dice = meta["dice"] if "dice" in meta else -1.0

    log(f"upload_model accessed by {username} - {meta['model_type']} - {model_path} - client dice {dice}")

    try:
        original_hash = meta["hash"]
    except:
        original_hash = None

    local_hash = calculate_file_hash(model_path)

    if original_hash is not None and original_hash != local_hash:
        log("Error during model upload")
        return {"message": "Communication error during upload"}, 500

    #model.dump(open(model_path, "wb"))

    print("Starting merge...")
    # Thread needed. With multiprocessing.Process this will block in docker+nginx
    # (daemon=True/False works both)
    # merge_thread = Thread(target=merge_model, args=(meta["model_type"], model_path), daemon=False)
    merge_thread = Thread(target=merge_model_thread, args=(meta["model_type"], model_path), daemon=False)
    merge_thread.start()

    merged_model = 1

    if merged_model is not None:
        return {"message": "upload successful"}, 200
    else:
        print("Info: merging of models failed, because validation Dice too low.")
        return {"message": "merging of models failed, because validation Dice too low."}, 500


@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Upload user data and save them to db/uploaded_data/<username>/<timestamp>.npz

    available data fields:
        api_key
    """
    meta = request.form.to_dict()
    if not valid_credentials(meta["api_key"]):
        log(f"Upload request of {meta['model_type']} rejected because api key {meta['api_key']} is invalid")
        return {"message": "invalid access code"}, 401

    username = get_username(meta["api_key"])

    data_dir = f"{DB_DIR}/uploaded_data/{username}"
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    request.files['data_binary'].save(f"{data_dir}/{int(time.time())}.npz")

    log(f"upload_data accessed by {username} - upload successful")
    return {"message": "upload successful"}, 200


def evaluate_model_thread(model_type, model_file):
    model = DynamicDLModel.Load(open(model_file, 'rb'))
    utils_evaluate_model(model_type, model, cleanup=True)
    del model
    gc.collect()


@app.route('/evaluate_model', methods=["POST"])  
def evaluate_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    username = get_username(meta["api_key"])
    log(f"evaluate_model accessed by {username} - {meta['model_type']} - {meta['timestamp']}")

    model = f"{MODELS_DIR}/{meta['model_type']}/{meta['timestamp']}.model"
    if not os.path.isfile(model):
        return {"message": "invalid model - not found"}, 500

    eval_thread = Thread(target=evaluate_model_thread, args=(meta["model_type"], model), daemon=False)
    eval_thread.start()

    return {"message": "starting evaluation successful"}, 200


if __name__ == '__main__':
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True)
