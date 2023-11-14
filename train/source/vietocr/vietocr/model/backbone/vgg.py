import torch
from torch import nn
from vietocr.model.backbone.vgg_torch import vgg19_bn_custom


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.1):
        super(Vgg, self).__init__()

        if name == "vgg11_bn":
            cnn = vgg_torch.vgg11_bn(pretrained=False)
        elif name == "vgg19_bn":
            cnn = vgg19_bn_custom(pretrained=False)

            # NOTE: image-net pretrained model off VGG19, removed some layers
            cnn.load_state_dict(
                torch.load(
                    "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/train/VietOCR/vgg19_cut.pth"
                )
            )

        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(
                    kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0
                )
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(0.1)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)
        # conv = conv.transpose(-1, -2)

        # convert torch.Size([1, 256, 2, 115]) to torch.Size([1, 256, 115, 2])
        conv = conv.permute(0, 1, 3, 2)
        conv = conv.flatten(2)

        # convert torch.Size([1, 256, 230]) to torch.Size([230, 1, 256])
        conv = conv.permute(2, 0, 1)

        return conv


def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg("vgg11_bn", ss, ks, hidden, pretrained, dropout)


def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg("vgg19_bn", ss, ks, hidden, pretrained, dropout)
