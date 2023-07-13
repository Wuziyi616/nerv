import copy

import cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch


def to_vis(img, scale=True):
    """Convert img to np.uint8 array that can be used to visualization."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError(f'Unrecognized type: {type(img)}')
    # we assume pixel values are either [0., 1.], or [-1., 1.] or [0, 255]
    if scale:
        if img.min() >= 0. and img.max() <= 1.:
            img = np.round(img * 255.).astype(np.uint8)
        elif img.min() >= -1. and img.max() <= 1.:
            img = np.round((img + 1.) / 2. * 255.).astype(np.uint8)
        else:
            img = np.round(img).astype(np.uint8)
    return img


def plot_imgs_seq(imgs, figsize=None, show=True):
    """Plot images in different figures."""
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [to_vis(img) for img in imgs]
    for i, img in enumerate(imgs):
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(f'{i}')

    if show:
        plt.show()


def plot_imgs_grid(imgs, grids=None, figsize=None, title=None, show=True):
    """Plot images in one figure using subplots."""
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [to_vis(img) for img in imgs]
    if grids is None:
        num_imgs = len(imgs)
        x = int(np.sqrt(num_imgs))
        y = int(np.ceil(num_imgs / x))
        grids = (x, y)
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(grids[0], grids[1], i + 1)
        plt.imshow(img)

    if title:
        plt.title(title)

    if show:
        plt.show()


FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_put_text_lines(img, label, org, color, fontScale=0.5, thickness=1):
    """Put text lines (splited with `\n`) on an image."""
    lines = label.split('\n')
    for i, line in enumerate(lines):
        (l_w, l_h_no_baseline), baseline = cv2.getTextSize(
            line,
            fontFace=FONT,
            fontScale=fontScale,
            thickness=thickness,
        )
        l_h = l_h_no_baseline + baseline + 2
        cv2.putText(
            img,
            line,
            org=org,
            fontFace=FONT,
            fontScale=fontScale,
            color=color,
            thickness=thickness)
        org = (org[0], org[1] + l_h)


def cv2_draw_bboxes(img, boxes, labels, colors, fontsize=0.5, thickness=1):
    """Draw bboxes on an image with labels and captions.

    This is much faster than torchvision.ops.draw_bounding_boxes
    """
    img = copy.deepcopy(img)  # avoid in-place modification

    if colors is None:
        colors = cv2.applyColorMap(
            np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        colors = [tuple(*item) for item in colors.tolist()]
    elif not isinstance(colors, (list, tuple)):
        colors = [colors] * len(boxes)

    if isinstance(colors[0], str):
        assert isinstance(colors, (list, tuple))
        colors = [
            tuple(int(x * 255) for x in matplotlib.colors.to_rgb(c))
            for c in colors
        ]  # convert str-based colors to (R, G, B) colors

    img = to_vis(img)
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i, 0]), int(boxes[i, 1]))
        pt2 = (int(boxes[i, 2]), int(boxes[i, 3]))
        label = labels[i]
        color = colors[i % len(colors)]
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness)
        cv2_put_text_lines(
            img,
            label,
            org=(pt1[0], pt1[1]),
            color=color,
            fontScale=fontsize,
            thickness=thickness)

    return img
