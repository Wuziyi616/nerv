import random
import numpy as np

import torch
import torch.distributed as dist


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


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if 'int' in str(array.dtype):
        assert 0 <= array.min() <= array.max() <= 255
    elif 'float' in str(array.dtype):
        assert 0. <= array.min() <= array.max() <= 1.
        array = np.round(array * 255.).astype(np.uint8)
    if is_video:
        return _convert4save_video(array)
    else:
        return _convert4save_img(array)
