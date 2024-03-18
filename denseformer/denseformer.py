import torch


class InPlaceSetSlice(torch.autograd.Function):

  @staticmethod
  def forward(ctx, full_tensor, last_slice, x_idx, x_val):
    full_tensor[x_idx] = x_val
    ctx.x_idx = x_idx
    ret = torch.Tensor().to(full_tensor.device)
    ret.set_(full_tensor[:x_idx + 1])
    return ret

  @staticmethod
  def backward(ctx, grad_out):
    if ctx.x_idx == 0:
      return None, None, None, grad_out[ctx.x_idx]
    else:
      return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
  full_tensor, last_slice = x_acc
  new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
  return full_tensor, new_slice


class DWAModules(torch.nn.Module):

  def __init__(self, n_blocks, dilation=1, period=1):
    super().__init__()
    self.n_blocks = n_blocks
    self.dilation = dilation
    self.period = period
    self.alphas = torch.nn.ModuleList([torch.nn.Linear((i+1+dilation)//dilation, 1, bias=False) if (i+1)%period == 0 else None for i in range(n_blocks)])
    self.accumulators = None
    self._init_weights()

  def _init_weights(self):
    for module in self.alphas:
      if module is not None:
        module.weight.data.zero_()
        module.weight.data[0, -1] = 1.

  def init_accumulators(self, x):
    x_accs = []
    for i in range(self.dilation):
      current_group_size = (self.n_blocks + 1) // self.dilation
      if i < (self.n_blocks + 1) % self.dilation:
        current_group_size += 1
      x_accs.append((torch.zeros((current_group_size, *x.shape), device=x.device, dtype=x.dtype), None))
    x_accs[0] = apply_inplace_set(x_accs[0], 0, x)
    self.accumulators = x_accs

  def forward(self, x, block_idx):
    assert self.accumulators is not None, "`init_accumulators(x)` needs to be called first"
    self.accumulators[(block_idx+1) % self.dilation] = apply_inplace_set(
        self.accumulators[(block_idx+1) % self.dilation], 
        (block_idx+1)//self.dilation,
        x  
    )
    if (block_idx+1) % self.period == 0:
      x = torch.tensordot(self.alphas[block_idx].weight.view(-1), self.accumulators[(block_idx+1)%self.dilation][1], dims=1)
    return x
