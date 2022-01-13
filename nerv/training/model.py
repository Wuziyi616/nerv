import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        pass

    def eval_loss_function(self, data_dict):
        """Loss computation in eval."""
        pass

    def loss_function(self, data_dict):
        if not self.training:
            return self.eval_loss_function(data_dict)
        pass

    def calc_train_loss(self):
        """Compute training loss."""
        pass

    @property
    def dtype(self):
        pass
