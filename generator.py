import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, depth=64, downsample=3, num_blocks=9):
        """
        Arguments:
            in_channels: an integer.
            out_channels: an integer.
            depth: an integer.
            downsample: an integer, the input will
                be downsampled in `2**downsample` times
                before applying resnet blocks.
            num_blocks: an integer, number of resnet blocks.
        """
        super(Generator, self).__init__()

        # BEGINNING

        start = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=7, bias=False, padding=3),
            nn.InstanceNorm2d(depth, affine=True),
            nn.ReLU(inplace=True)
        )

        # DOWNSAMPLING

        down_path = []
        for i in range(downsample):
            m = 2**i  # multiplier
            down_path.append(Downsample(depth * m))

        # MIDDLE BLOCKS

        blocks = []
        m = 2**downsample

        for _ in range(num_blocks):
            blocks.append(ResnetBlock(depth * m))

        # UPSAMPLING

        up_path = []
        for i in range(downsample):
            m = 2**(downsample - 1 - i)
            up_path.append(Upsample(depth * m * 2))

        # END

        self.end = nn.Sequential(
            nn.Conv2d(depth, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

        layers = [start] + down_path + blocks + up_path
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        I assume that h and w are
        divisible by 2**downsample.

        Input and output tensors represent
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """

        x = 2.0 * x - 1.0
        x = self.layers(x)

        x = self.end(x)
        x = 0.5 * x + 0.5

        return x


class Downsample(nn.Module):

    def __init__(self, d):
        super(Downsample, self).__init__()

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 1, 'bias': False
        }

        self.conv = nn.Conv2d(d, 2 * d, **params)
        self.norm = nn.InstanceNorm2d(2 * d, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, 2 * d, h / 2, w / 2].
        """

        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class Upsample(nn.Module):

    def __init__(self, d):
        super(Upsample, self).__init__()

        params = {
            'kernel_size': 3, 'stride': 1,
            'padding': 1, 'bias': False
        }

        self.conv = nn.Conv2d(d, d // 2, **params)
        self.norm = nn.InstanceNorm2d(d // 2, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d / 2, 2 * h, 2 * w].
        """

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ResnetBlock(nn.Module):

    def __init__(self, d):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(d, d, kernel_size=3, bias=False, padding=1)
        self.norm1 = nn.InstanceNorm2d(d, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(d, d, kernel_size=3, bias=False, padding=1)
        self.norm2 = nn.InstanceNorm2d(d, affine=True)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d, h, w].
        """

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.norm2(y)

        return x + y
