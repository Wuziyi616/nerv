import numpy as np


def hstack_array(arrs, pad=5):
    # each arr is of shape [..., H, W, 3]
    arr_shape = arrs[0].shape
    assert all(arr.shape == arr_shape for arr in arrs)
    W = arr_shape[-2]
    num_arrs = len(arrs)
    arr_shape = list(arr_shape)
    arr_shape[-2] = pad * (num_arrs - 1) + W * num_arrs
    stack_arr = np.zeros(arr_shape, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        start_idx = i * (W + pad)
        stack_arr[..., start_idx:start_idx + W, :] = arr
    return stack_arr


def vstack_array(arrs, pad=5):
    # each arr is of shape [..., H, W, 3]
    arr_shape = arrs[0].shape
    assert all(arr.shape == arr_shape for arr in arrs)
    H = arr_shape[-3]
    num_arrs = len(arrs)
    arr_shape = list(arr_shape)
    arr_shape[-3] = pad * (num_arrs - 1) + H * num_arrs
    stack_arr = np.zeros(arr_shape, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        start_idx = i * (H + pad)
        stack_arr[..., start_idx:start_idx + H, :, :] = arr
    return stack_arr
