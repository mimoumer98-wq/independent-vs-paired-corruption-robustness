import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResUNet(nn.Module):
    """ResNet-34 encoder (ImageNet-pretrained) with a U-Net-style decoder,
    for binary segmentation of grayscale images."""

    def __init__(self, in_channels=1, out_channels=1, pretrained=True, dropout_p=0.2):
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        if in_channels == 1:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride, padding=old_conv.padding, bias=False)
            if pretrained:
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc2 = resnet.layer1  # 64
        self.enc3 = resnet.layer2  # 128
        self.enc4 = resnet.layer3  # 256
        self.enc5 = resnet.layer4  # 512

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = self._dec_block(256 + 256, 256, dropout_p=0.3)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = self._dec_block(128 + 128, 128, dropout_p=0.25)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._dec_block(64 + 64, 64, dropout_p=0.2)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = self._dec_block(64 + 64, 64, dropout_p=0.2)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(32, out_channels, 1),
        )

    @staticmethod
    def _dec_block(in_ch, out_ch, dropout_p=0.2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.up5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e1r = F.interpolate(e1, size=d2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e1r], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        seg_logits = self.head(d1)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_logits
