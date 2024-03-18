# DenseFormer

This repository contains a helpful python package to implement DenseFormers as described in the paper: Enhancing Information Flow in Transformers via Depth Weighted Averaging. 

## Installation

The code is arranged as a `denseformer` package. To install the `denseformer` package, run:

```
pip install -e .
```

## Usage

The following shows how to transform a simplified Transformer class into a DenseFormer in only 3 steps:

```python
import torch
from denseformer import DWAModules 

class DenseFormer(torch.nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.dwa_modules = DWAModules(config.n_blocks, config.dilation, config.dwa_period) # Step 1
    self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
    self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
    self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
    self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight

  def forward(self, idx):
    x = self.wte(idx) 
    self.dwa_modules.init_accumulators(x) # Step 2
    for i in range(self.config.n_blocks):
      x = self.blocks[i](x)
      x = self.dwa_modules(x, block_idx=i) # Step 3
    x = self.ln_f(x)
    logits = self.lm_head(x)
    return logits
```

## Warning

The module use `nn.Linear` submodules for the DWA weights. If you force some initialization on all the `nn.Linear` submodules you might break the DWA initialization. Simply call `self.dwa_modules._init_weights()` again in that case.


