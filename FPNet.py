from torch import nn
import torch
from torchvision import models


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class FPNet18(nn.Module):
    def __init__(self, band_num, class_num):
        super(FPNet18, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.band_num_op = 13
        self.band_num_sa = 2
        self.name = 'FPNet18'

        channel = [64, 128, 256, 512]

        resnet_op = models.resnet18(pretrained=True)
        resnet_sa = models.resnet18(pretrained=True)

        self.firstConv_op = nn.Sequential(
            nn.Conv2d(self.band_num_op, channel[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )

        self.firstConv_sa = nn.Sequential(
            nn.Conv2d(self.band_num_sa, channel[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )

        self.maxpool_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_sa = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.e1_op = resnet_op.layer1
        self.e2_op = resnet_op.layer2
        self.e3_op = resnet_op.layer3
        self.e4_op = resnet_op.layer4

        self.e1_sa = resnet_sa.layer1
        self.e2_sa = resnet_sa.layer2
        self.e3_sa = resnet_sa.layer3
        self.e4_sa = resnet_sa.layer4

        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(channel[3], channel[2], 4, 2, 1),
            nn.Conv2d(channel[2], channel[2], 3, 1, 1)
        )
        self.c4 = nn.Conv2d(channel[2] * 2, channel[2], 3, 1, 1)

        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(channel[2], channel[1], 4, 2, 1),
            nn.Conv2d(channel[1], channel[1], 3, 1, 1)
        )
        self.c3 = nn.Conv2d(channel[1] * 2, channel[1], 3, 1, 1)

        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(channel[1], channel[0], 4, 2, 1),
            nn.Conv2d(channel[0], channel[0], 3, 1, 1)
        )
        self.c2 = nn.Conv2d(channel[0] * 2, channel[0], 3, 1, 1)

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(channel[0], channel[0], 4, 2, 1),
            nn.Conv2d(channel[0], channel[0], 3, 1, 1)
        )
        self.c1 = nn.Conv2d(channel[0] * 2, channel[0], 3, 1, 1)

        self.d0 = nn.Sequential(
            nn.ConvTranspose2d(channel[0], channel[0] // 2, 4, 2, 1),
            nn.Conv2d(channel[0] // 2, self.band_num_op, 3, 1, 1)
        )

    def forward(self, x):
        optical = x[:, 2:, :, :]
        sar = x[:, 0:2, :, :]
        cloud = x[:, 2:, :, :]

        conv1_op = self.firstConv_op(optical)
        pooling_op = self.maxpool_op(conv1_op)
        e1_op = self.e1_op(pooling_op)
        e2_op = self.e2_op(e1_op)
        e3_op = self.e3_op(e2_op)
        e4_op = self.e4_op(e3_op)

        conv1_sa = self.firstConv_sa(sar)
        pooling_sa = self.maxpool_sa(conv1_sa)
        e1_sa = self.e1_sa(pooling_sa)
        e2_sa = self.e2_sa(e1_sa)
        e3_sa = self.e3_sa(e2_sa)
        e4_sa = self.e4_sa(e3_sa)

        d4 = self.d4(e4_op + e4_sa)
        c4 = self.c4(torch.cat((d4, e3_sa + e3_op), 1))  # 4, 256, 16, 16

        d3 = self.d3(c4)
        c3 = self.c3(torch.cat((d3, e2_sa + e2_op), 1))  # 4, 128, 32, 32

        d2 = self.d2(c3)
        c2 = self.c2(torch.cat((d2, e1_sa + e1_op), 1))  # [4, 64, 64, 64]

        d1 = self.d1(c2)
        c1 = self.c1(torch.cat((d1, conv1_op + conv1_sa), 1))

        d0 = self.d0(c1)

        return d0 + cloud


class FPNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(FPNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.band_num_op = 13
        self.band_num_sa = 2
        self.name = 'FPNet'

        channel = [64, 128, 256, 512]

        resnet_op = models.resnet34(pretrained=True)
        resnet_sa = models.resnet34(pretrained=True)

        self.firstConv_op = nn.Sequential(
            nn.Conv2d(self.band_num_op, channel[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )

        self.firstConv_sa = nn.Sequential(
            nn.Conv2d(self.band_num_sa, channel[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )

        self.maxpool_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_sa = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.e1_op = resnet_op.layer1
        self.e2_op = resnet_op.layer2
        self.e3_op = resnet_op.layer3
        self.e4_op = resnet_op.layer4

        self.e1_sa = resnet_sa.layer1
        self.e2_sa = resnet_sa.layer2
        self.e3_sa = resnet_sa.layer3
        self.e4_sa = resnet_sa.layer4

        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(channel[3], channel[2], 4, 2, 1),
            nn.Conv2d(channel[2], channel[2], 3, 1, 1)
        )
        self.c4 = nn.Conv2d(channel[2] * 2, channel[2], 3, 1, 1)

        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(channel[2], channel[1], 4, 2, 1),
            nn.Conv2d(channel[1], channel[1], 3, 1, 1)
        )
        self.c3 = nn.Conv2d(channel[1] * 2, channel[1], 3, 1, 1)

        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(channel[1], channel[0], 4, 2, 1),
            nn.Conv2d(channel[0], channel[0], 3, 1, 1)
        )
        self.c2 = nn.Conv2d(channel[0] * 2, channel[0], 3, 1, 1)

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(channel[0], channel[0], 4, 2, 1),
            nn.Conv2d(channel[0], channel[0], 3, 1, 1)
        )
        self.c1 = nn.Conv2d(channel[0] * 2, channel[0], 3, 1, 1)

        self.d0 = nn.Sequential(
            nn.ConvTranspose2d(channel[0], channel[0] // 2, 4, 2, 1),
            nn.Conv2d(channel[0] // 2, self.band_num_op, 3, 1, 1)
        )

    def forward(self, x):
        optical = x[:, 2:, :, :]
        sar = x[:, 0:2, :, :]
        cloud = x[:, 2:, :, :]

        conv1_op = self.firstConv_op(optical)
        pooling_op = self.maxpool_op(conv1_op)
        e1_op = self.e1_op(pooling_op)
        e2_op = self.e2_op(e1_op)
        e3_op = self.e3_op(e2_op)
        e4_op = self.e4_op(e3_op)

        conv1_sa = self.firstConv_sa(sar)
        pooling_sa = self.maxpool_sa(conv1_sa)
        e1_sa = self.e1_sa(pooling_sa)
        e2_sa = self.e2_sa(e1_sa)
        e3_sa = self.e3_sa(e2_sa)
        e4_sa = self.e4_sa(e3_sa)

        d4 = self.d4(e4_op + e4_sa)
        c4 = self.c4(torch.cat((d4, e3_sa + e3_op), 1))  # 4, 256, 16, 16

        d3 = self.d3(c4)
        c3 = self.c3(torch.cat((d3, e2_sa + e2_op), 1))  # 4, 128, 32, 32

        d2 = self.d2(c3)
        c2 = self.c2(torch.cat((d2, e1_sa + e1_op), 1))  # [4, 64, 64, 64]

        d1 = self.d1(c2)  # 64 128 128
        c1 = self.c1(torch.cat((d1, conv1_op + conv1_sa), 1))

        d0 = self.d0(c1)

        return d0 + cloud


if __name__ == '__main__':
    num_classes = 13
    in_batch, inchannel, in_h, in_w = 4, 15, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)

    net = FPNet(inchannel, num_classes)
    out = net(x)
    print(out.shape)
