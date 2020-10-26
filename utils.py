import os
import glob
import datetime

import pydicom as dicom
import numpy as np

# hide tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set to 2 to hide all warnings

from dl.DynamicDLModel import DynamicDLModel


MODELS_DIR = "models"
TEST_DATA_DIR = "test_data"

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
    lines = open("api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return True
    return False


def get_username(api_key):
    lines = open("api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return username
    return None


def get_model_types():
    available_models = [f for f in glob.glob(f"{MODELS_DIR}/*")]
    return [m.split("/")[-1] for m in available_models]


def get_models(model_type):
    available_models = [f for f in glob.glob(f"{MODELS_DIR}/{model_type}/*.model")]
    return sorted([m.split("/")[-1].split(".")[0] for m in available_models])


def evaluate_model(model_type: str, model: DynamicDLModel) -> float:
    
    for file in glob.glob(f"{TEST_DATA_DIR}/{model_type}/*"):
        arr, res, _ = load_dicom_file(file)
        seg = model.apply({"image": arr, "resolution": res[:2]})
        print(seg.keys())
        print(seg["VL"].shape)

        # todo: compare to groundtruth

    return 1.0


def merge_model(model_type, new_model_path, dice_thr=0.8):
    latest_timestamp = get_models(model_type)[-1]
    latest_model = DynamicDLModel.Load(open(f"{MODELS_DIR}/{model_type}/{latest_timestamp}.model", 'rb'))
    new_model = DynamicDLModel.Load(open(new_model_path, 'rb'))

    if evaluate_model(model_type, new_model) < dice_thr: return None

    # todo: is this the right usage of apply_delta?
    merged_model = latest_model.apply_delta(new_model)

    if evaluate_model(model_type, merged_model) < dice_thr: return None
    return merged_model


def log(text):
    with open("log.txt", "a") as f:
        f.write(str(datetime.datetime.now()) + " " + text + "\n")


if __name__ == '__main__':
    # valid_credentials("abc123")
    # print(get_model_types())
    # print(get_models("thigh"))
    # log("hello world")

    # f, res, hdr = load_dicom_file("test_data/thigh/thigh_test.dcm")
    # print(res)

    model = DynamicDLModel.Load(open(f"{MODELS_DIR}/thigh/1603281013.model", 'rb'))
    r = evaluate_model("thigh", model)

