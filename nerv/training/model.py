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
        return self.calc_train_loss(data_dict, out_dict)

    def loss_function(self, data_dict):
        """General warpper for loss calculation."""
        out_dict = self.forward(data_dict)
        if self.training:
            out_dict = self.calc_train_loss(data_dict, out_dict)
        else:
            out_dict = self.calc_eval_loss(data_dict, out_dict)
        # batch_size for statistics accumulation
        for v in data_dict.values():
            if isinstance(v, torch.Tensor):
                out_dict['batch_size'] = v.shape[0]
                break
        return out_dict

    @property
    def dtype(self):
        pass

    @property
    def device(self):
        pass
