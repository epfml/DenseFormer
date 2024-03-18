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
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import logging
import time

from tqdm import tqdm

import config
import models
from data.utils import get_dataset, prepare_dataset
from optim.base import train_base
import distributed
from optim.utils import get_batch

def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=none_or_str, required=True)
    parser.add_argument('--config_format', type=str, required=False)
    
    args, rem_args = parser.parse_known_args()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            args.checkpoint, args.checkpoint_filename = os.path.split(args.checkpoint)
        else:
            args.checkpoint_filename = "ckpt.pt"

        with open(os.path.join(args.checkpoint, "summary.json")) as f:
            summary = json.load(f)

        for k, v in summary['args'].items():
            if k == "config_format" and args.config_format is not None:
                continue
            if k not in ["device", "dtype"]:
                setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=argparse.ArgumentParser(allow_abbrev=False), args=rem_args, namespace=args)


def get_as_batch(data, seq_length, batch_size, device='cpu', sample_size=None):
    all_ix = list(range(0, len(data), seq_length))
    assert all_ix[-1] + seq_length + 1 > len(data)
    all_ix.pop()
    if sample_size is not None:
        all_ix = np.random.choice(all_ix, size=sample_size // seq_length, replace=False).tolist()
    
    idx = 0
    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, iterations, acc_steps, batch_size, sequence_length, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {}

    num_substeps_per_epoch = len(data['val']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        # torchdynamo.config.log_level = logging.DEBUG
        model = torch.compile(model) # requires pytorch 2.0+

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []
    seq_length = extra_args.eval_seq_length or extra_args.sequence_length
    with torch.no_grad():
        torch.set_printoptions(sci_mode=False)
        t0 = time.time()
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=extra_args.device, 
                    sample_size=extra_args.eval_sample_size
                )
            ),
            total=iceildiv(
                extra_args.eval_sample_size // seq_length if extra_args.eval_sample_size is not None else 
                iceildiv(len(data['val']), seq_length), 
                batch_size
            )
        ):
            cnt = 0
            with type_ctx:
                outputs = model(x, targets=y, get_logits=True)
        
            val_loss = outputs['loss']
            acc = ((outputs['logits'].argmax(-1) == y).float().mean())
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
        t1 = time.time()
        dt = t1 - t0
        eval_per_batch_time = dt * 1000 / len(acc_list)
        

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['eval_per_batch_time'] = eval_per_batch_time

    return stats

def main(args): 


    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")

    if distributed_backend.is_master_process():
        prepare_dataset(args)
    distributed_backend.sync()
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x}, strict=False)

    model = distributed_backend.transform_model(model)
    
    print(f"\Evaluating model={args.model} \n{vars(args)}\n")

    stats = evaluate(model, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  distributed_backend=distributed_backend,
                  extra_args=args)

    print(stats)
    
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
