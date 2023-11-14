import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import models


class MobileNetV3(nn.Module):
    def __init__(self, hidden, pretrained=True, dropout=0.5):
        super(MobileNetV3, self).__init__()

        cnn = models.mobilenet_v3_large(pretrained=pretrained)

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(960, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        conv = conv.permute(0, 1, 3, 2)
        conv = conv.flatten(2)
        conv = conv.permute(2, 0, 1)

        return conv


class Swin_V2_B(nn.Module):
    def __init__(self, hidden, pretrained=True, dropout=0.5):
        super(Swin_V2_B, self).__init__()

        swin = models.swin_v2_t(weights="Swin_V2_T_Weights.IMAGENET1K_V1")
        self.features = swin.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(1024, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        # [1, 256, 15, 2]
        conv = conv.permute(0, 1, 3, 2)

        # use permute instead of transpose
        conv = conv.flatten(2)
        conv = conv.permute(2, 0, 1)

        return conv


class EfficientNet(nn.Module):
    def __init__(self, hidden, pretrained=True, dropout=0.5):
        super(EfficientNet, self).__init__()

        self.features = models.efficientnet_b0(
            weights="EfficientNet_B0_Weights.IMAGENET1K_V1"
        ).features

        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(1280, hidden, 1)


    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        # [1, 256, 15, 2]
        conv = conv.permute(0, 1, 3, 2)

        # use permute instead of transpose
        conv = conv.flatten(2)
        conv = conv.permute(2, 0, 1)

        return conv
