import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter


########################################################################################
# Gabor Net
class GaborNet(nn.Module):
    """
    Conditional GaborNet based neural field.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layers.
        code_size (int): Size of the code layer.
        out_size (int): Size of the output layer.
        n_layers (int): Number of layers in the model.
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, alpha=6.0, beta=1.0, **kwargs):
        super().__init__()
        self.film_layers = nn.ModuleList(
            [FiLM(in_size, code_size, hidden_size)] +
            [FiLM(hidden_size, code_size, hidden_size) for _ in range(int(n_layers))]
        )
        self.output_layer = nn.Linear(hidden_size, out_size)
        self.filters = nn.ModuleList([GaborLayer(
            in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1), alpha / (n_layers + 1), beta) for _ in range(n_layers + 1)])

    def forward(self, x, code):
        out = self.filters[0](x) * self.film_layers[0](torch.zeros((code.size(0), *x.shape[1:])).to(code.device), code)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.film_layers[i](out, code)
        return self.output_layer(out)


class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_scale = weight_scale

        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )

    def forward(self, x):
        exponent = -0.5 * ((x ** 2).sum(-1)[..., None] + (self.mu ** 2).sum(-1)[None, :] - 2 * x @ self.mu.T) * self.gamma[None, :]
        exponent = torch.exp(exponent)
        pre_act = F.linear(x, self.weight * self.weight_scale)
        return torch.cat([torch.sin(pre_act) * exponent, torch.cos(pre_act) * exponent], dim=-1)


class FiLM(nn.Module):
    """
    A PyTorch module implementing the Feature-wise Linear Modulation (FiLM) layer.

    Args:
        in_h_features (int): Number of input features for the h tensor.
        in_z_features (int): Number of input features for the z tensor.
        out_features (int): Number of output features.

    Attributes:
        in_h_features (int): Number of input features for the h tensor.
        in_z_features (int): Number of input features for the z tensor.
        out_features (int): Number of output features.
        Wp (torch.nn.Parameter): Weight parameter for the shift operation.
        Ws (torch.nn.Parameter): Weight parameter for the scale operation.
        W (torch.nn.Parameter): Weight parameter for the linear transformation.
        bias (torch.nn.Parameter): Bias parameter.

    Methods:
        reset_parameters():
            Initializes the parameters of the layer.

        forward(_h: Tensor=None, _z: Tensor=None) -> Tensor:
            Forward pass of the FiLM layer.
            Args:
                _h (Tensor): Input tensor with shape (batch, height, width, depth, coord_dim) or (batch, aggregated, coord_dim).
                _z (Tensor): Input tensor with shape (batch, code_dim).
            Returns:
                Tensor: Output tensor after applying FiLM operations.
    """

    def __init__(self, in_h_features: int, in_z_features: int, out_features: int) -> None:
        super(FiLM, self).__init__()
        self.in_h_features = in_h_features
        self.in_z_features = in_z_features
        self.out_features = out_features
        self.Wp = Parameter(torch.empty(out_features, in_z_features))
        self.Ws = Parameter(torch.empty(out_features, in_z_features))
        self.W = Parameter(torch.empty(out_features, in_h_features))
        self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_h_features)
        init.kaiming_uniform_(self.Wp, a=math.sqrt(5))
        init.kaiming_uniform_(self.Ws, a=math.sqrt(5))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)


    def forward(self, _h: Tensor=None, _z: Tensor=None) -> Tensor:
        res = torch.matmul(_h, self.W.t())
        scale = torch.matmul(_z, self.Ws.t())
        shift = torch.matmul(_z, self.Wp.t())
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        while scale.dim() < res.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        res = scale * res + shift + self.bias
        return res
