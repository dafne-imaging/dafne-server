#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import gc
import sys
import os
import re
import time

from dafne_dl.DynamicDLModel import DynamicDLModel
from dafne_dl.DynamicTorchModel import DynamicTorchModel
from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel
from utils import evaluate_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', metavar='model', type=str, help='Model file')
parser.add_argument('--type', '-t', dest='model_type', metavar='type', type=str, required=False,
                    help='Type of model (optional, otherwise inferred by the path)')
parser.add_argument('--test-path', '-p', dest='test_path', metavar='path', type=str, required=False,
                    help='Path to npz test files (optional)')
parser.add_argument('--model_class', '-c', type=str, help='model class (i.e. DynamicDLModel)')

args = parser.parse_args()

model_path = os.path.abspath(args.model)

test_path = args.test_path

model_class_name = args.model_class

MODEL_CLASSES = {
    "DynamicDLModel": DynamicDLModel,
    "DynamicTorchModel": DynamicTorchModel,
    "DynamicEnsembleModel": DynamicEnsembleModel,
}

model_class = MODEL_CLASSES.get(model_class_name, DynamicDLModel)


if test_path is not None:
    model_type_or_dir = test_path
else:
    model_type_or_dir = args.model_type

    if model_type_or_dir is None:
        m = re.search(r'/models/([^/_]+)', model_path)
        if not m:
            print('Invalid path to model - cannot determine the model type')
            sys.exit(1)

        model_type_or_dir = m.group(1)
print('Model type or dir:', model_type_or_dir)

def run_evaluation():
    print('Loading model...')
    # model = DynamicEnsembleModel.Load(open(model_path, 'rb'))
    # model = DynamicDLModel.Load(open(model_path, 'rb'))
    model = model_class.Load(open(model_path, 'rb'))

    print('Evaluating model...')
    t = time.time()
    dice = evaluate_model(model_type_or_dir, model, save_log=False)
    elapsed = time.time() - t
    print('Dice score:', dice)
    print('Elapsed time', elapsed)

run_evaluation()

# @profile
# def run():
#     print("Run 1")
#     run_evaluation()
#     gc.collect()
#     print("after gc")
#     print("Run 2")
#     run_evaluation()
#     gc.collect()
#     print("Run 3")
#     run_evaluation()
#     gc.collect()
#
# run()
