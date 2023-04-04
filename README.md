# NERV

<p align="center"><img src="src/NERV.jpg" alt="NERV" width="800"/></p>

> God's in his heaven. All's right with the world!
> — Pippa Passes

> Graduate's in his heaven. All's right with the code!
> — The dream of a struggling PhD student

Personal Python toolbox including project templates, useful functions, etc.
The training framework is tailored for the computing cluster at [Vector Institute](https://vectorinstitute.ai/), but can also be applied to other platforms.

## Credit

Greatly inspired by and lots of code borrowed from:

-   [cvbase](https://github.com/hellock/cvbase)
-   [utils3d](https://github.com/Steve-Tod/utils3d)
-   [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

\* Image credit: [Neon Genesis Evangelion](https://en.wikipedia.org/wiki/Neon_Genesis_Evangelion)

## Installation

First manually install PyTorch with cuda support (see `requirements.txt` for versions we tested), then run `pip install -e .` to install the whole package.

### Possible Issues

-   When initializing the Trainer, we check the number of GPUs [here](https://github.com/Wuziyi616/nerv/blob/e83ac66c6ce30e1ca3d0a287df9d3699ed9ec499/nerv/training/method.py#L117).
    If you use `CUDA_VISIBLE_DEVICES=0,... python train.py xxx` to launch training in the commandline, then you will pass the check.
    But if you set `os.environ['CUDA_VISIBLE_DEVICES'] = '0,...'` in the python file (after importing `nerv`), this may trigger an error.
    We recommend to always use `CUDA_VISIBLE_DEVICES=0,...` to run python commands.

-   Some users have encountered a weird version issue with `opencv`.
    For `nerv-v0.1.0`, `opencv-python==4.5.5.64`, `4.5.3.56` and `4.6.0.66` are three tested versions.

## Related Projects

-   [SlotFormer](https://github.com/pairlab/SlotFormer) (ICLR'23)
