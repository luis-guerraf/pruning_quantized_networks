import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import BinarizeLinear, BinarizeConv2d, PrunableBatchNorm2d, PrunableBatchNorm1d, PrunableLinear

class VGG11_binary(nn.Module):

    def __init__(self, num_classes=10, FLReal=False):
        super(VGG11_binary, self).__init__()
        self.infl_ratio = 1

        self.features = nn.Sequential(
            BinarizeConv2d(3, 64*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PrunableBatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PrunableBatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            PrunableBatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PrunableBatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            PrunableBatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PrunableBatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512 * self.infl_ratio, 512 * self.infl_ratio, kernel_size=3, padding=1, bias=True),
            PrunableBatchNorm2d(512 * self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512 * self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PrunableBatchNorm2d(512),
            nn.Hardtanh(inplace=True)
        )

        # If first and last layer real
        if FLReal:
            self.classifier = nn.Sequential(
                PrunableLinear(512 * 1 * 1, num_classes, bias=True)
            )
        else:
            self.classifier = nn.Sequential(
                BinarizeLinear(512 * 1 * 1, num_classes, bias=True),
                nn.BatchNorm1d(num_classes, affine=False)
            )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999), 'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

        self.ReTrain = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999), 'lr': 5e-3},
            1: {'lr': 1e-3},
            2: {'lr': 5e-4},
            3: {'lr': 1e-4},
            4: {'lr': 5e-5}
        }


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def vgg11_binary(**kwargs):
    dataset, FLReal = map(
        kwargs.get, ['dataset', 'FLReal'])

    num_classes = 10 if dataset == 'cifar10' else 100

    return VGG11_binary(num_classes, FLReal)
