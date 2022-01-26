import numpy as np

from torch.nn import LayerNorm, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def conv_out_shape(in_size, stride, padding, kernel_size, dilation=1):
    """Calculate the output shape of a Conv layer."""
    return np.floor((in_size + 2 * padding - dilation *
                     (kernel_size - 1) - 1) / float(stride) + 1)


def deconv_out_shape(in_size,
                     stride,
                     padding,
                     kernel_size,
                     out_padding,
                     dilation=1):
    """Calculate the output shape of a ConvTranspose layer."""
    return (in_size - 1) * stride - 2 * padding + dilation * (
        kernel_size - 1) + out_padding + 1


def filter_wd_parameters(model, skip_list=()):
    """Create parameter groups for optimizer.

    We do two things:
        - filter out params that do not require grad
        - exclude bias and Norm layers
    """
    # all params that require grad
    all_params = set(filter(lambda p: p.requires_grad, model.parameters()))

    no_decay = set()
    # exclude bias
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or name in skip_list:
            no_decay.add(param)
    # exclude norm weight
    for m in model.modules():
        if isinstance(m, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            no_decay.add(m.weight)
    decay = all_params - no_decay

    return {'decay': list(decay), 'no_decay': list(no_decay)}
