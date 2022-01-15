import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        pass

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        pass

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        pass

    def loss_function(self, data_dict):
        """General warpper for loss calculation."""
        out_dict = self.forward(data_dict)
        if self.training:
            out_dict = self.calc_train_loss(data_dict, out_dict)
        else:
            out_dict = self.calc_eval_loss(data_dict, out_dict)
        # batch_size for statistics accumulation
        out_dict['batch_size'] = list(data_dict.values())[0].shape[0]
        return out_dict

    @property
    def dtype(self):
        pass

    @property
    def device(self):
        pass
