# inspired by https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEma(nn.Module):
    """
    ModelEma is a class that maintains an exponential moving average (EMA) of a given model's parameters.

    Attributes:
        module (nn.Module): A copy of the input model used to accumulate the moving average of weights.
        decay (float): The decay rate for the EMA. Default is 0.9999.

    Methods:
        update(model):
            Updates the EMA parameters using the current parameters of the given model.

        set(model):
            Sets the EMA parameters to be exactly the same as the given model's parameters.
    """
    def __init__(self, model, decay=0.9999):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.to(model.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
