import os, glob, time, io
from threading import Thread

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from markupsafe import escape

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set to 2 to hide all warnings

from dl.LocalModelProvider import LocalModelProvider
from dl.DynamicDLModel import DynamicDLModel
from utils import valid_credentials, get_model_types, get_models, get_username, merge_model, log
from utils import MODELS_DIR

app = Flask(__name__)


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
    return {"latest_timestamp": latest_timestamp}, 200


@app.route('/get_model', methods=["POST"])  
def get_model():
    meta = request.json
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    # todo: read and write access only to models dir
    model = f"{MODELS_DIR}/{meta['model_type']}/{meta['timestamp']}.model"
    if not os.path.isfile(model):
        return {"message": "invalid model - not found"}, 500
    model = DynamicDLModel.Load(open(model, "rb"))
    model_bytes = model.dumps()
    username = get_username(meta["api_key"])
    log(f"get_model accessed by {username} - {meta['model_type']} - {meta['timestamp']}")
    return send_file(io.BytesIO(model_bytes), mimetype='image/jpg'), 200


@app.route('/upload_model', methods=['POST'])
def upload_model():
    meta = request.form.to_dict()
    if not valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401
    if meta["model_type"] not in get_model_types():
        return {"message": "invalid model type"}, 500
            
    username = get_username(meta["api_key"])
    model_binary = request.files['model_binary'].read()  # read() is needed to get bytes from FileStorage object
    model_delta = DynamicDLModel.Loads(model_binary)
    model_orig_path = f"{MODELS_DIR}/{meta['model_type']}/{model_delta.timestamp_id}.model"
    model_orig = DynamicDLModel.Load(open(model_orig_path, "rb"))

    # calc_delta(): new_model - orig_model = delta
    # apply_delta(): model_orig + delta = model_new

    # Model delta: only activate if rest of federated learning working properly
    # model = model_orig.apply_delta(model_delta)
    model = model_delta

    model_path = f"{MODELS_DIR}/{meta['model_type']}/uploads/{str(int(time.time()))}_{username}.model"
    model.dump(open(model_path, "wb"))

    log(f"upload_model accessed by {username} - {meta['model_type']} - {model_path}")

    print("Starting merge...")
    # todo: set this higher
    dice_thr = -0.1
    # merged_model = merge_model(meta["model_type"], model_path)
    thread = Thread(target=merge_model, args=(meta["model_type"], model_path))
    thread.daemon = True
    thread.start()
    merged_model = 1

    if merged_model is not None:
        return {"message": "upload successful"}, 200
    else:
        print("Info: merging of models failed, because validation Dice too low.")
        return {"message": "merging of models failed, because validation Dice too low."}, 500



def my_async_func(a):
    print("Inside async: starting")
    print(f"Inside async: param: {a}")
    time.sleep(4)    
    print("Inside async: ending")
    return 5

@app.route('/my_test', methods=["POST"])
def my_test():
    meta = request.json
    print("Inside my_test()")
    
    print("Calling async thread...")
    thread = Thread(target=my_async_func, args=("what",))
    thread.daemon = True
    thread.start()

    print("Inside my_test(): returning")
    return {"latest_timestamp": 123}, 200


if __name__ == '__main__':
    # Only for debugging while developing
    app.run(debug=True)
