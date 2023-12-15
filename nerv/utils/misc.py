import os
import time
import random
import numpy as np

import torch

from .tensor import to_numpy


def datetime2str(form='%Y-%m-%d_%H-%M-%S'):
    datetime = time.strftime(form, time.localtime())
    return datetime


def sort_file_by_time(files):
    """Sort filenames by its created time."""
    # the first one is the oldest (earliest created)
    return sorted(files, key=lambda x: os.path.getmtime(x))


def assert_array_shape(xyz, shapes=()):
    """Check array shape.

    Args:
        xyz (np.ndarray): array
        shape (tuple of tuple of ints, optional): possible target shapes,
            -1 means arbitrary. Defaults to ((-1, 3)).
    """
    if not shapes:
        raise ValueError('"shapes" cannot be empty')

    if isinstance(shapes[0], int):
        shapes = (shapes, )

    flags = {x: True for x in range(len(shapes))}
    for idx, shape in enumerate(shapes):
        if len(xyz.shape) != len(shape):
            flags[idx] = False

        for dim, num in enumerate(shape):
            if num == -1:
                continue
            elif xyz.shape[dim] != num:
                flags[idx] = False
    if sum(flags.values()) == 0:  # None of the possible shape works
        raise ValueError(
            f'Input array {xyz.shape} is not in target shapes {shapes}!')


def array_equal(a, b):
    """Compare if two arrays are the same.

    Args:
        a/b: can be np.ndarray or torch.Tensor.
    """
    if a.shape != b.shape:
        return False
    try:
        assert (a == b).all()
        return True
    except:
        return False


def array_in_list(array, lst):
    """Check whether an array is in a list."""
    for v in lst:
        if array_equal(array, v):
            return True
    return False


def get_key_from_value(d, v):
    """Get the key of a value in a dict."""
    k = [k for k, v_ in d.items() if v_ == v]
    if k > 1:
        raise ValueError('Duplicate values in the dict!')
    elif k == 1:
        return k[0]
    return None


def set_seed(seed=1, deterministic=False):
    """Set the random seed of the environment for reproducibility.

    Args:
        seed (int): the random seed to set.
        deterministic (bool, optional): whether to use deterministic torch
            backend. Default: False.
    """
    print('Using random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        print('Using deterministic pytorch backends')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _convert4save_img(array):
    """Convert a image array to be saveable."""
    # extend channel axis
    if len(array.shape) == 2:
        array = np.stack([array] * 3, axis=-1)
    # [C, H, W] --> [H, W, C], where C == 3
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    return np.ascontiguousarray(array)


def _convert4save_video(array):
    """Convert a video array to be saveable."""
    # extend channel axis
    if len(array.shape) == 3:
        array = np.stack([array] * 3, axis=-1)
    # [T, C, H, W] --> [T, H, W, C], where C == 3
    if array.shape[1] == 3:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def convert4save(array, is_video=False):
    """Check the dtype and value range of input array for save.

    Need to convert to [(T), H, W, C] with np.uint8 value range [0, 255].

    Args:
        array (np.ndarray or torch.Tensor): array to be converted.
        is_video (bool, optional): whether the array is a video or image.
            Default: False (means array is an image).

    Returns:
        np.ndarray: the converted array ready for save (image or video).
    """
    array = to_numpy(array)
    if 'int' in str(array.dtype):
        assert 0 <= array.min() <= array.max() <= 255
    elif 'float' in str(array.dtype):
        assert 0. <= array.min() <= array.max() <= 1.
        array = np.round(array * 255.).astype(np.uint8)
    if is_video:
        return _convert4save_video(array)
    else:
        return _convert4save_img(array)


def timeit(func, iters=10, *args, **kwargs):
    """Measure the time of a function."""
    start = time.time()
    for _ in range(iters):
        _ = func(*args, **kwargs)
    end = time.time()
    return (end - start) / iters
