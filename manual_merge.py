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

from argparse import ArgumentParser
import json
import os
from dafne_dl.DynamicDLModel import DynamicDLModel
from dafne_dl.DynamicTorchModel import DynamicTorchModel
from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db', 'server_config.json')

if __name__ == '__main__':
    parser = ArgumentParser("Manually merge multiple deep learning models")
    parser.add_argument('base_model', metavar='Model1', type=str, help='Base model')
    parser.add_argument('new_models', metavar='ModelN', type=str, nargs='+', help='Models to merge')
    parser.add_argument('model_class', type=str, help='model class (i.e. DynamicDLModel)')
    parser.add_argument('-o', dest='output_path', metavar='path', type=str, required=True, help='Output folder')

    args = parser.parse_args()

    config = json.load(open(config_file))

    MODEL_CLASSES = {
        "DynamicDLModel": DynamicDLModel,
        "DynamicTorchModel": DynamicTorchModel,
        "DynamicEnsembleModel": DynamicEnsembleModel,
    }

    base_model = args.base_model
    new_model_list = args.new_models
    output_dir = args.output_path
    model_class_name = args.model_class

    model_class = MODEL_CLASSES.get(model_class_name, DynamicDLModel)

    print(f"model_class {model_class}")

    original_model_weight = config['original_model_weight']

    print('Original model weight setting', original_model_weight)

    # merged_model = DynamicDLModel.Load(open(base_model, 'rb'))
    merged_model = model_class.Load(open(base_model, 'rb'))
    print(f'merged_model {merged_model}')

    for new_model in new_model_list:
        print('Loading', new_model)
        # new_model = DynamicDLModel.Load(open(new_model, 'rb'))
        new_model = model_class.Load(open(new_model, 'rb'))
        print(f'new_model {new_model}')
        merged_model = merged_model*original_model_weight + new_model*(1-original_model_weight)

    merged_model.reset_timestamp()
    merged_model.dump(open(os.path.join(output_dir, f'{merged_model.timestamp_id}.model'), 'wb'))
