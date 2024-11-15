import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        level_channels=[32, 64, 128],
        bottleneck_channel=1024,
        smaller=False
    ):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(level_channels)):
            in_ch = in_channels if i == 0 else level_channels[i - 1]
            out_ch = level_channels[i]
            self.enc_blocks.append(
                Conv3DBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bottleneck=False,
                    smaller=smaller
                )
            )
        self.bottleNeck = Conv3DBlock(
            in_channels=out_ch,
            out_channels=bottleneck_channel,
            bottleneck=True,
            smaller=smaller
        )
        self.fc = nn.Linear(bottleneck_channel, bottleneck_channel)

    def forward(self, voxels):
        # encoder
        out = voxels
        for block in self.enc_blocks:
            out, _ = block(out)
        out, _ = self.bottleNeck(out)

        # pooling
        out = torch.nn.functional.avg_pool3d(out, out.size()[2:])
        out = out.squeeze()
        out = self.fc(out)

        return out


class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, non_linearity=True):
        super(SingleConv3D, self).__init__()

        self.use_bn = use_bn
        self.non_linearity = non_linearity

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )

        if use_bn:
            self.bn = nn.BatchNorm3d(num_features=out_channels)

        if non_linearity:
            self.nl = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        if self.use_bn:
            x = self.bn(x)
        if self.non_linearity:
            x = self.nl(x)
        return x


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck=False,
        use_bn=True,
        res_block=True,
        smaller=False,
    ):
        super(Conv3DBlock, self).__init__()
        self.res_block = res_block

        # first conv
        if smaller:
            level_channels_in = [
                in_channels,
                out_channels,
                out_channels // 2,
            ]
            level_channels_out = [
                out_channels,
                out_channels // 2,
                out_channels,
            ]
        else:
            level_channels_in = [
                out_channels,
                out_channels // 2,
                out_channels // 2,
                out_channels // 2,
                out_channels,
            ]
        self.conv_layers = nn.ModuleList()
        for i in range(len(level_channels_in)):
            if smaller:
                self.conv_layers.append(
                    SingleConv3D(
                        in_channels=level_channels_in[i],
                        out_channels=level_channels_out[i],
                        use_bn=use_bn,
                        non_linearity=(i != len(level_channels_out) - 1),
                    )
                )
            else:
                in_ch = in_channels if i == 0 else level_channels_in[i - 1]
                out_ch = level_channels_in[i]
                self.conv_layers.append(
                    SingleConv3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        use_bn=use_bn,
                        non_linearity=(i != len(level_channels_in) - 1),
                    )
                )

        # non linearity
        self.nl = nn.ReLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = input
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(res)
                res = x.clone()
            else:
                res = conv(res)

        if self.res_block:
            res += x

        res = self.nl(res)

        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res

        return out, res
