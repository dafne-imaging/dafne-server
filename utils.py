import os
import glob
import datetime
from collections import defaultdict
import time
from pathlib import Path
import json

import pydicom as dicom
import numpy as np
import nibabel as nib
from tqdm import tqdm

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set to 2 to hide all warnings

from dl.DynamicDLModel import DynamicDLModel


MODELS_DIR = "db/models"
TEST_DATA_DIR = "db/test_data"

# def load_dicom_file(fname):
#     ds = dicom.read_file(fname)
#     # rescale dynamic range to 0-4095
#     try:
#         pixelData = ds.pixel_array.astype(np.float32)
#     except:
#         ds.decompress()
#         pixelData = ds.pixel_array.astype(np.float32)
#     ds.PixelData = ""
#     return pixelData, ds


def load_dicom_file(fname):
    print(fname)
    ds = dicom.read_file(fname)
    # rescale dynamic range to 0-4095
    try:
        pixelData = ds.pixel_array.astype(np.float32)
    except:
        ds.decompress()
        pixelData = ds.pixel_array.astype(np.float32)

    try:
        slThickness = ds.SpacingBetweenSlices
    except:
        slThickness = ds.SliceThickness

    ds.PixelData = ""
    resolution = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(slThickness)]
    return pixelData, resolution, ds


def valid_credentials(api_key):
    lines = open("db/api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return True
    return False


def get_username(api_key):
    lines = open("db/api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return username
    return None


def get_model_types():
    available_models = list(glob.glob(f"{MODELS_DIR}/*"))
    return [m.split("/")[-1] for m in available_models]


def get_models(model_type):
    available_models = list(glob.glob(f"{MODELS_DIR}/{model_type}/*.model"))
    return sorted([m.split("/")[-1].split(".")[0] for m in available_models])


def delete_older_models(model_type, keep_latest=5):
    """
    Only keep the newest N models inside of model_type/* and model_type/uploads/*
    """

    # # Remove model_type/*
    # available_models = list(glob.glob(f"{MODELS_DIR}/{model_type}/*.model"))
    # available_models = sorted([m.split("/")[-1].split(".")[0] for m in available_models])
    # if len(available_models) > keep_latest:
    #     for model in available_models[:-keep_latest]:
    #         rm_path = Path(MODELS_DIR) / model_type / f"{model}.model"
    #         print(f"removing: {rm_path}")
    #         os.remove(rm_path)

    # Remove model_type/uploads/*
    available_models = list(glob.glob(f"{MODELS_DIR}/{model_type}/uploads/*.model"))
    available_models = sorted([m.split("/")[-1].split("_")[0] for m in available_models])
    if len(available_models) > keep_latest:
        for model in available_models[:-keep_latest]:
            rm_path = list((Path(MODELS_DIR) / model_type / "uploads").glob(f"{model}_*.model"))[0]
            print(f"removing: {rm_path}")
            os.remove(rm_path)


def my_f1_score(y_true, y_pred):
    """
    Binary f1. Same results as sklearn f1 binary.

    y_true: 1D / 2D / 3D array
    y_pred: 1D / 2D / 3D array
    """
    intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def evaluate_model(model_type: str, model: DynamicDLModel) -> float:
    
    if model_type != "Thigh":
        print("WARNING: Validation data only available for thigh model. Skipping validation!")
        return 1.0

    for file in glob.glob(f"{TEST_DATA_DIR}/*.nii.gz"):
        print(f"Processing subject: {str(file).split('/')[-1]}")
        # data, res, _ = load_dicom_file(file)
        img = nib.load(file)
        res = img.header.get_zooms()
        data = img.get_fdata()

        scores = defaultdict(list)
        slices = range(data.shape[2])

        #todo: remove this to use all slices
        print("WARNING: Only evaluating on a subset of slices for faster runtime.")
        slices = [20,30]

        for idx in tqdm(slices):
            slice = data[:, :, idx]
            pred = model.apply({"image": slice, "resolution": res[:2]})
            gt = np.load(file.replace(".nii.gz", ".npz"))
            
            pred_keys = pred.keys()
            gt_keys = gt.files
            
            if set(pred_keys) != set(gt_keys):
                raise ValueError(f"Keys of prediction and groundtruth are different: " +
                                 f"{pred_keys} vs {gt_keys}")

            for key in pred_keys:
                dice = my_f1_score(gt[key][:, :, idx], pred[key])
                scores[key].append(dice)

    scores = {k: np.array(v).mean() for k, v in scores.items()}
    print(scores)
    mean_score = np.array(list(scores.values())).mean()
    log(f"evaluate_model.mean_score: {mean_score}", p=True)
    return mean_score


def merge_model(model_type, new_model_path):
    """
    This will take a (weighted) average of the weights of two models.
    If the new_model or the resulting merged model have a lower validation dice score
    than dice_thr then the merged model will be discarded. Otherwise it will become the
    new default model.
    """
    print("Merging...")
    config = json.load(open("db/server_config.json"))

    latest_timestamp = get_models(model_type)[-1]
    latest_model = DynamicDLModel.Load(open(f"{MODELS_DIR}/{model_type}/{latest_timestamp}.model", 'rb'))
    new_model = DynamicDLModel.Load(open(new_model_path, 'rb'))

    # Check that model_ids are identical
    if latest_model.model_id != new_model.model_id:
        print(f"WARNING: Model_IDs do not match. Can not merge models. " +
              f"({latest_model.model_id} vs {new_model.model_id})")
        return None

    # Validate dice of uploaded model
    if evaluate_model(model_type, new_model) < config["dice_threshold"]: 
        print("Score is below threshold. Returning None")
        return None

    # todo: is this the right usage of apply_delta?
    merged_model = latest_model.apply_delta(new_model)

    # Validate dice of merged model
    if evaluate_model(model_type, merged_model) < config["dice_threshold"]: 
        print("Score is below threshold. Returning None")
        return None

    print("Saving merged model as new main model...")
    new_model_path = f"{MODELS_DIR}/{model_type}/{str(int(time.time()))}.model"
    merged_model.dump(open(new_model_path, 'wb'))
    log(f"Saved merged model with timestamp: {merged_model.timestamp_id}", p=True)

    print("Deleting old models...")
    delete_older_models(model_type, keep_latest=config["nr_models_to_keep"])

    return merged_model


def log(text, p=False):
    if p:
        print(text)
    with open("db/log.txt", "a") as f:
        f.write(str(datetime.datetime.now()) + " " + text + "\n")


if __name__ == '__main__':
    ####### For testing #######
    model = DynamicDLModel.Load(open(f"{MODELS_DIR}/thigh/1603281013.model", 'rb'))
    r = evaluate_model("thigh", model)
