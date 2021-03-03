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

from dl.DynamicDLModel import DynamicDLModel, IncompatibleModelError
# from dl.labels.thigh import long_labels_split as thigh_labels
# from dl.labels.leg import long_labels_split as leg_labels
from dl.labels.thigh import long_labels as thigh_labels
from dl.labels.leg import long_labels as leg_labels
from dl.labels.thigh import short_labels as thigh_labels_short
from dl.misc import calc_dice_score


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

def keras_weighted_average(lhs: DynamicDLModel, rhs: DynamicDLModel, lhs_weight = 0.5):
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        average = lhs_weights[depth]*lhs_weight + rhs_weights[depth]*(1-lhs_weight)
        newWeights.append(average)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj

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


def _get_nonzero_slices(mask):
    slices = []
    for idx in range(mask.shape[2]):
        if mask[:,:,idx].max() > 0:
            slices.append(idx)
    return slices


def evaluate_model(model_type: str, model: DynamicDLModel, save_log=True, comment='') -> float:
    """
    This will evaluate model on all subjects in TEST_DATA_DIR/model_type.
    Per subject all slices which have ground truth annotations will be evaluated (only a subset of all slices
    per subject).
    """
    labels = leg_labels.values() if model_type.startswith("Leg") else thigh_labels.values()

    t = time.time()

    for file in Path(TEST_DATA_DIR).glob(f"{model_type}/*.npz"):
        print(f"Processing subject: {file.name}")
        img = np.load(file)
        slices_idxs = _get_nonzero_slices(img[f"mask_{list(labels)[0]}"])
        scores = defaultdict(list)
        for idx in tqdm(slices_idxs):
            pred = model.apply({"image": img["data"][:, :, idx],
                                "resolution": np.abs(img["resolution"][:2]),
                                "split_laterality": False})
            for label in labels:
                gt = img[f"mask_{label}"][:, :, idx]
                nr_voxels = gt.sum()
                dice = calc_dice_score(gt, pred[label])
                scores[label].append([dice, nr_voxels])

    scores_per_label = {k: np.array(v)[:, 0].mean() for k, v in scores.items()}
    print(scores_per_label)
    scores_flat = []
    for key, val in scores.items():
        for elem in val:
            scores_flat.append(elem)
    
    scores_flat = np.array(scores_flat)
    mean_score = np.average(scores_flat[:, 0], weights=scores_flat[:, 1])
    elapsed = time.time() - t
    if save_log:
        log(f"evaluating model {model_type}/{model.timestamp_id}.model: Dice: {mean_score:.6f} (time: {elapsed:.2}) {comment})", p=True)
        log_dice_to_csv(f"{model_type}/{model.timestamp_id}.model", mean_score)
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
        log(f"WARNING: Model_IDs do not match. Can not merge models. " +
              f"({latest_model.model_id} vs {new_model.model_id})", True)
        return

    # Validate dice of uploaded model
    if evaluate_model(model_type, new_model, comment='(Uploaded model)') < config["dice_threshold"]:
        log("Score of new model is below threshold.", True)
        return

    # The following is only valid if we are applying a difference between two models. However, we are sending a full model
    # merged_model = latest_model.apply_delta(new_model)

    #merged_model = keras_weighted_average(latest_model, new_model, lhs_weight=ORIGINAL_MODEL_WEIGHT)
    # The following is slower but more general (not limited to keras models, using the internal multiplication/sum functionality)
    original_weight = config["original_model_weight"]
    merged_model = latest_model*original_weight + new_model*(1-original_weight)

    merged_model.reset_timestamp()

    # Validate dice of merged model
    if evaluate_model(model_type, merged_model, comment='(Merged model)') < config["dice_threshold"]:
        log("Score of the merged model is below threshold.")
        return

    print("Saving merged model as new main model...")
    new_model_path = f"{MODELS_DIR}/{model_type}/{merged_model.timestamp_id}.model"
    temp_model_path = new_model_path + '.tmp'
    merged_model.dump(open(temp_model_path, 'wb')) # write to a tmp file to avoid serving an incompletely written model
    os.rename(temp_model_path, new_model_path)
    log(f"Saved merged model with timestamp: {merged_model.timestamp_id}", p=True)

    print("Deleting old models...")
    delete_older_models(model_type, keep_latest=config["nr_models_to_keep"])

    return


def log(text, p=False):
    if p:
        print(text)
    with open("db/log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} {text}\n")

def log_dice_to_csv(model_name, dice, comment=''):
    with open("db/dice.csv", "a") as f:
        f.write(f"{datetime.datetime.now()};{model_name};{dice:.6f};{comment}\n")

if __name__ == '__main__':
    ####### For testing #######
    model = DynamicDLModel.Load(open(f"{MODELS_DIR}/Thigh/1610001000.model", 'rb'))
    r = evaluate_model("Thigh", model)
