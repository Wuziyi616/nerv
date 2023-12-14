from typing import Union, List, Tuple

import numpy as np

import torch
import torch as th


def np_th_stack(values: List[Union[np.ndarray, th.Tensor]], axis: int = 0):
    """Stack a list of numpy arrays or tensors."""
    if isinstance(values[0], np.ndarray):
        return np.stack(values, axis=axis)
    elif isinstance(values[0], th.Tensor):
        return torch.stack(values, dim=axis)
    else:
        raise ValueError(f'Unknown type {type(values[0])}')


def np_th_concat(values: List[Union[np.ndarray, th.Tensor]], axis: int = 0):
    """Concat a list of numpy arrays or tensors."""
    if isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=axis)
    elif isinstance(values[0], th.Tensor):
        return torch.cat(values, dim=axis)
    else:
        raise ValueError(f'Unknown type {type(values[0])}')


def get_bbox_coords(bbox: Union[np.ndarray, th.Tensor], last4: bool = None):
    """Get the 4 coords (xyxy/xywh) from a bbox array or tensor."""
    if isinstance(bbox, list):
        bbox = np_th_stack(bbox, axis=0)
    if last4 is None:  # infer from shape, buggy when bbox.shape == (4, ..., 4)
        assert not bbox.shape[0] == bbox.shape[-1], f'Ambiguous {bbox.shape=}'
        if bbox.shape[0] == 4:
            last4 = False
        elif bbox.shape[-1] == 4:
            last4 = True
        else:
            raise ValueError(f'Unknown shape {bbox.shape}')
    if last4:
        a, b, c, d = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    else:
        a, b, c, d = bbox
    return (a, b, c, d), last4


def construct_bbox(abcd: Tuple[Union[np.ndarray, th.Tensor]], last4: bool):
    """Construct a bbox from 4 coords (xyxy/xywh)."""
    if last4:
        return np_th_stack(abcd, axis=-1)
    return np_th_stack(abcd, axis=0)


def xywh2xyxy(
        xywh: Union[np.ndarray, th.Tensor],
        format_: str = 'center',  # format of the source xywh bbox
        last4: bool = None):
    """Convert bounding box from xywh to xyxy format."""
    (x, y, w, h), last4 = get_bbox_coords(xywh, last4=last4)

    if format_ == 'center':
        x1, x2 = x - w / 2., x + w / 2.
        y1, y2 = y - h / 2., y + h / 2.
    elif format_ == 'corner':
        x1, x2 = x, x + w
        y1, y2 = y, y + h
    else:
        raise NotImplementedError(f'Unknown format {format_}')

    return construct_bbox((x1, y1, x2, y2), last4=last4)


def xyxy2xywh(
        xyxy: Union[np.ndarray, th.Tensor],
        format_: str = 'center',  # format of the target xywh bbox
        last4: bool = None):
    """Convert bounding box from xyxy to xywh format."""
    (x1, y1, x2, y2), last4 = get_bbox_coords(xyxy, last4=last4)

    w, h = x2 - x1, y2 - y1
    if format_ == 'center':
        x, y = (x1 + x2) / 2., (y1 + y2) / 2.
    elif format_ == 'corner':
        x, y = x1, y1
    else:
        raise NotImplementedError(f'Unknown format {format_}')

    return construct_bbox((x, y, w, h), last4=last4)


def batch_iou_xywh(
        bbox1: Union[np.ndarray, th.Tensor],
        bbox2: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    """
    Computes IoU between two [x,y,w,h,(cls_id)] **center** format bboxes,
      both bbox are in shape (N, 4/5) where N is the number of bboxes.
    If class_id is provided, take it into account by ignoring the IoU between
      bboxes of different classes.
    """
    bbox1 = bbox1[:, None]  # (M, 1, 4/5)
    bbox2 = bbox2[None]  # (1, N, 4/5)

    # define the element-wise min/max function for np or th
    if isinstance(bbox1, np.ndarray):
        assert isinstance(bbox2, np.ndarray)
        max_fn = np.maximum
        min_fn = np.minimum
    else:
        assert isinstance(bbox1, th.Tensor) and isinstance(bbox2, th.Tensor)
        max_fn = th.maximum
        min_fn = th.minimum

    x1y1 = max_fn(bbox1[..., :2] - bbox1[..., 2:4] / 2.,
                  bbox2[..., :2] - bbox2[..., 2:4] / 2.)
    x2y2 = min_fn(bbox1[..., :2] + bbox1[..., 2:4] / 2.,
                  bbox2[..., :2] + bbox2[..., 2:4] / 2.)
    wh = max_fn(0., x2y2 - x1y1)
    intersect = wh[..., 0] * wh[..., 1]
    union = bbox1[..., 2] * bbox1[..., 3] \
        + bbox2[..., 2] * bbox2[..., 3] \
        - intersect
    o = intersect / union

    # set IoU of between different class objects to 0
    if bbox1.shape[-1] == 5 and bbox2.shape[-1] == 5:
        o[bbox2[..., 4] != bbox1[..., 4]] = 0.

    return o


def batch_iou_xyxy(
        bbox1: Union[np.ndarray, th.Tensor],
        bbox2: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    """
    Computes IoU between two [x1,y1,x2,y2,(cls_id)] format bboxes,
      both bbox are in shape (N, 4/5) where N is the number of bboxes.
    If class_id is provided, take it into account by ignoring the IoU between
      bboxes of different classes.
    """
    bbox1 = bbox1[:, None]  # (M, 1, 4/5)
    bbox2 = bbox2[None]  # (1, N, 4/5)

    # define the element-wise min/max function for np or th
    if isinstance(bbox1, np.ndarray):
        assert isinstance(bbox2, np.ndarray)
        max_fn = np.maximum
        min_fn = np.minimum
    else:
        assert isinstance(bbox1, th.Tensor) and isinstance(bbox2, th.Tensor)
        max_fn = th.maximum
        min_fn = th.minimum

    x1y1 = max_fn(bbox1[..., :2], bbox2[..., :2])
    x2y2 = min_fn(bbox1[..., 2:4], bbox2[..., 2:4])
    wh = max_fn(0., x2y2 - x1y1)
    intersect = wh[..., 0] * wh[..., 1]
    union = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) \
        + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) \
        - intersect
    o = intersect / union

    # set IoU of between different class objects to 0
    if bbox1.shape[-1] == 5 and bbox2.shape[-1] == 5:
        o[bbox2[..., 4] != bbox1[..., 4]] = 0.

    return o
