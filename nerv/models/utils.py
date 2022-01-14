import numpy as np


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
