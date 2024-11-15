import torch
from torch import nn


class MLPResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 32,
        dropout: float = 0.1,
        bias_free: bool = False,
    ):
        super().__init__()

        # first norm + conv layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU() if not bias_free else nn.ReLU()
        self.mlp1 = nn.Linear(in_channels, out_channels, bias=not bias_free)

        # second norm + conv layer
        self.norm2 = nn.GroupNorm(n_groups, in_channels)
        self.act2 = nn.SiLU() if not bias_free else nn.ReLU()
        self.mlp2 = nn.Linear(out_channels, out_channels, bias=not bias_free)
        self.mlp2.weight.data.zero_()
        if not bias_free:
            self.mlp2.bias.data.zero_()

        if in_channels != out_channels:
            # self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.shortcut = nn.Linear(in_channels, out_channels, bias=not bias_free)
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.mlp1(h)

        h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.mlp2(h)

        return h + self.shortcut(x)


class MLPResCode(nn.Module):
    def __init__(
        self,
        code_dim: int = 1024,
        n_hidden_units: int = 2048,
        num_blocks: int = 4,
        n_groups: int = 32,
        dropout: float = 0.1,
        bias_free: bool = False,
        out_dim: int = None
    ):
        super().__init__()

        self.projection = nn.Linear(code_dim, n_hidden_units, bias=not bias_free)

        # encoder
        enc = []
        for i in range(num_blocks):
            enc.append(MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free))
        self.enc = nn.ModuleList(enc)

        # bottleneck
        self.middle = MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free)

        # decoder
        dec = []
        for i in reversed(range(num_blocks)):
            dec.append(MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free))
        self.dec = nn.ModuleList(dec)

        self.norm = nn.GroupNorm(n_groups, n_hidden_units)
        self.act = nn.SiLU() if not bias_free else nn.ReLU()
        self.final = nn.Linear(n_hidden_units, code_dim if out_dim is None else out_dim, bias=not bias_free)
        self.final.weight.data.zero_()
        if not bias_free:
            self.final.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)
        x = self.projection(x)

        # encoder
        hidden = [x]
        for m in self.enc:
            x = m(x)
            hidden.append(x)

        # bottleneck
        x = self.middle(x)

        # decoder
        for dec in self.dec:
            hid = hidden.pop()
            x = torch.add(x, hid)
            x = dec(x)

        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.act(x)
        x = self.final(x)

        return x
