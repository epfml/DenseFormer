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

import os
import zipfile
import urllib
import numpy as np
import tiktoken
import torch
import regex
import multiprocessing
import itertools
import functools


PG19_ORIGINAL_PATH = "./data/pg19"


def get_path(config):
    dataset_name = f"pg19"
    return os.path.join(os.path.dirname(__file__), f"datasets/{dataset_name}/")

def _read_directory(path):
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt") and filename[:-4].isnumeric():
            print(filename)
            with open(os.path.join(path, filename), 'r') as f:
                texts.append(f.read())
    return "<|endoftext|>".join(texts)
    
def prepare_pg19_data(config):
    DATA_PATH = get_path(config)
    print(DATA_PATH)
    if not os.path.exists(os.path.join(DATA_PATH, 'train.bin')) or not os.path.exists(os.path.join(DATA_PATH, 'val.bin')):
        os.makedirs(DATA_PATH, exist_ok=True)

        train_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(PG19_ORIGINAL_PATH, 'validation.bin'), dtype=np.uint16, mode='r')


        raw_tokenized_train = train_data
        raw_tokenized_eval = val_data

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16) 
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        train_tokenized.tofile(os.path.join(DATA_PATH, 'train.bin'))
        eval_tokenized.tofile(os.path.join(DATA_PATH, 'val.bin'))
        print("completed the tokenization process!")


def get_pg19_data(config):
    DATA_PATH = get_path(config)

    """ Inspired from https://github.com/tysam-code/hlb-gpt """
    
    train_data = np.memmap(os.path.join(DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
