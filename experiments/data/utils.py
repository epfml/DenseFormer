# Copyright 2023 Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from . import pg19, openwebtext2

PREPARE_GET_DATASET_MAP = {
    "pg19": (pg19.prepare_pg19_data, pg19.get_pg19_data),
    "owt2": (openwebtext2.prepare_openwebtext2_data, openwebtext2.get_openwebtext2_data)
}


def prepare_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][0](args)

def get_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][1](args)
