import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        """Forward function."""
        pass

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        pass

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        return self.calc_train_loss(data_dict, out_dict)

    def loss_function(self, data_dict):
        """General warpper for loss calculation."""
        assert not self.training, \
            '`model.calc_train_loss()` should be called outside the model. ' \
            'This change is introduced in nerv v0.4.0 to ensure gradient ' \
            'synchronization across GPUs in DDP training.' \
            'Refer to https://github.com/Wuziyi616/nerv/issues/1 for details.'
        out_dict = self.forward(data_dict)
        out_dict = self.calc_eval_loss(data_dict, out_dict)
        # batch_size for statistics accumulation
        for v in data_dict.values():
            if isinstance(v, torch.Tensor):
                out_dict['batch_size'] = v.shape[0]
                break
        return out_dict

    def _training_epoch_start(self, method=None):
        """Things to do at the beginning of every training epoch."""
        pass

    def _training_step_start(self, method=None):
        """Things to do at the beginning of every training step."""
        pass

    def _training_step_end(self, method=None):
        """Things to do at the end of every training step."""
        pass

    def _training_epoch_end(self, method=None):
        """Things to do at the end of every training epoch."""
        pass

    def load_weight(self, ckp_path, strict=True):
        """Load checkpoint from a file.

        Args:
            ckp_path (str): Path to checkpoint file.
            strict (bool, optional): Whether to allow different params for
                the model and checkpoint. Defaults to True.
        """
        ckp = torch.load(ckp_path, map_location='cpu')
        if 'state_dict' in ckp:
            ckp = ckp['state_dict']
        self.load_state_dict(ckp, strict=strict)

    @property
    def dtype(self):
        pass

    @property
    def device(self):
        pass
