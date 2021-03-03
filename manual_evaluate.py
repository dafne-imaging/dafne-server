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

import sys
import os
import re
import time

from dl.DynamicDLModel import DynamicDLModel
from utils import evaluate_model

model_path = os.path.abspath(sys.argv[1])
m = re.search(r'/models/([^/_]+)', model_path)
if not m:
    print('Invalid path to model - cannot determine the model type')
    sys.exit(1)

model_type = m.group(1)
print('Model type:', model_type)

print('Loading model...')
model = DynamicDLModel.Load(open(model_path, 'rb'))

print('Evaluating model...')
t = time.time()
dice = evaluate_model(model_type, model, save_log=False)
elapsed = time.time() - t
print('Dice score:', dice)
print('Elapsed time', elapsed)
