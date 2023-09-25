import numpy as np

import torch

from torchmetrics import MeanMetric as TorchMeanMetric
from torchmetrics.aggregation import BaseAggregator


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, device=torch.device('cpu')):
        self.reset()
        self.device = device

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        if np.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return torch.tensor(self.avg).to(self.device)

    def to(self, device):
        self.device = device
        return self


class MeanMetric(BaseAggregator):
    """My implemented MeanMetric class inspired by torchmetrics.MeanMetric."""

    def __init__(
        self,
        device,
        nan_strategy="warn",
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        self.to(device)
        super().__init__(
            "sum",
            torch.tensor(0.0).to(device),
            nan_strategy=nan_strategy,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state(
            "weight",
            default=torch.tensor(0.0).to(device),
            dist_reduce_fx="sum",
        )
        self.to(device)

    def update(self, value, weight=1.0):
        return TorchMeanMetric.update(self, value, weight)

    def compute(self):
        """Compute the aggregated value in torch.Tensor format."""
        if (self.weight == 0.).item():
            return torch.tensor(0.).to(self.device)
        return self.value / self.weight

    def to(self, device):
        self._device = device
        return self
