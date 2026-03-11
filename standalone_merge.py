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
from utils import merge_model, log


if __name__ == '__main__':
    parser = ArgumentParser("Wrapper around utils.merge_model which can be called using subprocess.call")
    parser.add_argument('model_type', type=str, help='model type')
    parser.add_argument('model_class', type=str, help='model class (i.e. DynamicDLModel)')
    parser.add_argument('new_model_path', type=str, help='path of the new model we want to merge')
    args = parser.parse_args()

    log("Using standalone merger", p=True)
    merge_model(args.model_type, args.model_class, args.new_model_path)
