import gc
import json
import os, time
import shutil
from threading import Thread
import subprocess
from typing import Union

import uvicorn
from fastapi import FastAPI, Form, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set to 2 to hide all warnings

from dafne_dl.LocalModelProvider import LocalModelProvider
from dafne_dl.DynamicDLModel import DynamicDLModel
from dafne_dl.DynamicTorchModel import DynamicTorchModel
from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel
from utils import valid_credentials, get_model_types, get_models, get_username, merge_model, log
from utils import evaluate_model as utils_evaluate_model
from utils import MODELS_DIR
from dafne_dl.misc import calculate_file_hash

def _save_file(file_obj, path):
    try:
        with open(path, 'wb') as f:
            shutil.copyfileobj(file_obj, f)
    except IOError:
        print("IOError")
        os.remove(path)
    finally:
        file_obj.close()

MODEL_CLASSES = {
    "DynamicDLModel": DynamicDLModel,
    "DynamicTorchModel": DynamicTorchModel,
    "DynamicEnsembleModel": DynamicEnsembleModel,
}

app = FastAPI()

# On the server /mnt/data/dafne-server-db will be mounted to db when starting the docker container
DB_DIR = "db"

# when using request.post(url, json=dict)
class APIAndModelRequestJSON(BaseModel):
    api_key: str
    model_type: str = ''
    timestamp: str = ''
    message: str = ''

@app.post('/get_available_models')
def get_available_models(response: Response, api_key_json: APIAndModelRequestJSON):
    api_key = api_key_json.api_key
    if not valid_credentials(api_key):
        response.status_code = 401
        return {"message": "invalid access code"}

    return {"models": get_model_types()}


@app.post('/info_model')
def info_model(response: Response, request_json: APIAndModelRequestJSON):
    if not valid_credentials(request_json.api_key):
        response.status_code = 401
        return {"message": "invalid access code"}

    timestamps = get_models(request_json.model_type)

    hashes = {}
    for stamp in timestamps:
        model_path = f"{MODELS_DIR}/{request_json.model_type}/{stamp}.model"
        model_hash = calculate_file_hash(model_path, True)
        hashes[stamp] = model_hash

    #print(timestamps)
    #print(hashes)

    out_dict = {"latest_timestamp": timestamps[-1], "hash": hashes[timestamps[-1]], "hashes": hashes, "timestamps": timestamps}
    json_file_path = f"{MODELS_DIR}/{request_json.model_type}/model.json"
    if os.path.exists(json_file_path):
        # add the content of the json file to the dictionary
        out_dict.update(json.load(open(json_file_path, "rb")))

    return out_dict


@app.post('/get_model')
def get_model(response: Response, request_json: APIAndModelRequestJSON):
    if not valid_credentials(request_json.api_key):
        response.status_code = 401
        return {"message": "invalid access code"}

    # todo: read and write access only to models dir
    model = f"{MODELS_DIR}/{request_json.model_type}/{request_json.timestamp}.model"
    if not os.path.isfile(model):
        response.status_code = 500
        return {"message": "invalid model - not found"}
    username = get_username(request_json.api_key)
    log(f"get_model accessed by {username} - {request_json.model_type} - {request_json.timestamp}")
    return FileResponse(model, headers={'mimetype': 'application/octet-stream'})


def merge_model_thread(model_type, model_class, new_model_path):
    subprocess.call(f"python standalone_merge.py {model_type} {model_class} {new_model_path}", shell=True)

@app.post('/upload_model')
def upload_model(response: Response, api_key: str = Form(...),
                 model_type: str = Form(...),
                 original_hash: Union[str, None] = Form(None),
                 dice: Union[float,None] = Form(None),
                 model_binary: UploadFile = UploadFile(...)):
    """
    available data fields:
        api_key
        model_type
        dice (optional)
    """
    if not valid_credentials(api_key):
        log(f"Upload request of {model_type} rejected because api key {api_key} is invalid")
        response.status_code = 401
        return {"message": "invalid access code"}

    username = get_username(api_key)

    if model_type not in get_model_types():
        log(f"Upload request of {model_type} from {username} rejected because model type is invalid")
        response.status_code = 500
        return {"message": "invalid model type"}

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
    model_path = f"{MODELS_DIR}/{model_type}/uploads/{str(int(time.time()))}_{username}.model"
    _save_file(model_binary.file, model_path)

    log(f"upload_model accessed by {username} - {model_type} - {model_path} - client dice {dice}")



    local_hash = calculate_file_hash(model_path)

    if original_hash is not None and original_hash != local_hash:
        log("Error during model upload")
        return {"message": "Communication error during upload"}, 500

    #model.dump(open(model_path, "wb"))

    print("Starting merge...")
    # Thread needed. With multiprocessing.Process this will block in docker+nginx
    # (daemon=True/False works both)
    # merge_thread = Thread(target=merge_model, args=(meta["model_type"], model_path), daemon=False)
    
    json_file_path = f"{MODELS_DIR}/{model_type}/model.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            model_info = json.load(f)
            model_class_name = model_info.get("model_type", "DynamicDLModel")
    else:
        model_class_name = "DynamicDLModel"
    
    model_class = MODEL_CLASSES.get(model_class_name, DynamicDLModel)

    merge_thread = Thread(target=merge_model_thread, args=(model_type, model_class, model_path), daemon=False)
    merge_thread.start()

    return {"message": "upload successful"}



@app.post('/upload_data')
def upload_data(response: Response, api_key: str = Form(...),
                 data_binary: UploadFile = UploadFile(...)):
    """
    Upload user data and save them to db/uploaded_data/<username>/<timestamp>.npz

    available data fields:
        api_key
    """
    if not valid_credentials(api_key):
        log(f"Data upload request rejected because api key {api_key} is invalid")
        response.status_code = 401
        return {"message": "invalid access code"}

    username = get_username(api_key)

    data_dir = f"{DB_DIR}/uploaded_data/{username}"
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    _save_file(data_binary.file, f"{data_dir}/{str(int(time.time()))}.npz")

    log(f"upload_data accessed by {username} - upload successful")
    return {"message": "upload successful"}


def evaluate_model_thread(model_type, model_file):
    # model = DynamicDLModel.Load(open(model_file, 'rb'))
    # utils_evaluate_model(model_type, model, cleanup=True)
    # del model
    # gc.collect()
    json_file_path = f"{MODELS_DIR}/{model_type}/model.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            model_info = json.load(f)
            model_class_name = model_info.get("model_type", "DynamicDLModel")
    else:
        model_class_name = "DynamicDLModel"
    
    model_class = MODEL_CLASSES.get(model_class_name, DynamicDLModel)
    model = model_class.Load(open(model_file, "rb"))
    utils_evaluate_model(model_type, model, cleanup=True)

    del model
    gc.collect()


@app.post('/evaluate_model')
def evaluate_model(response: Response, request_json: APIAndModelRequestJSON):
    if not valid_credentials(request_json.api_key):
        response.status_code = 401
        return {"message": "invalid access code"}

    # username = get_username(api_key)
    username = get_username(request_json.api_key)
    log(f"evaluate_model accessed by {username} - {request_json.model_type} - {request_json.timestamp}")
    
    model = f"{MODELS_DIR}/{request_json.model_type}/{request_json.timestamp}.model"

    if not os.path.isfile(model):
        response.status_code = 500
        return {"message": "invalid model - not found"}

    eval_thread = Thread(target=evaluate_model_thread, args=(request_json.model_type, model), daemon=False)
    eval_thread.start()

    return {"message": "starting evaluation successful"}

@app.post('/log')
def log_message(response: Response, request_json: APIAndModelRequestJSON):
    """
    Log a message from a user

    available data fields:
        api_key
        message
    """
    if not valid_credentials(request_json.api_key):
        response.status_code = 401
        return {"message": "invalid access code"}

    username = get_username(request_json.api_key)
    message = request_json.message
    log(f"Log message from {username} - {message}", True)
    return {'message': 'ok'}


if __name__ == '__main__':
    # Only for debugging while developing
    uvicorn.run('main:app', host='0.0.0.0', port=5000)
    # command line: uvicorn main:app --port 5000
