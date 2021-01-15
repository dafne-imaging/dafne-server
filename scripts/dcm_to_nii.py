import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) # go up 1 level
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import glob, time, io
from pathlib import Path

import numpy as np
import nibabel as nib

from utils import load_dicom_file
from utils import MODELS_DIR


if __name__ == '__main__':
    
    dir_in = Path(sys.argv[1]).absolute()
    file_out = Path(sys.argv[2]).absolute()
    # if len(sys.argv > 3):
    #     npz_in = Path(sys.argv[3]).absolute()
    # else:
    #     npz_in = None

    files = list(dir_in.glob("*.dcm"))

    img = []
    for file in files:
        arr, _, _ = load_dicom_file(file)
        img.append(arr)
    img = np.array(img).transpose(1,2,0)
    print(f"shape: {img.shape}")
    arr, res, ds = load_dicom_file(list(files)[0])

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = res[i]

    nib.save(nib.Nifti1Image(img, affine), file_out)

    # if npz_in:
    #     masks_out = []
    #     masks = np.load(npz_in)
    #     labels = masks.files
    #     for label in labels:
    #         masks_out.append(masks[label])

    
