import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d, PrunableBatchNorm2d, PrunableBatchNorm1d, PrunableConv2d, PrunableLinear

__all__ = ['resnet_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = PrunableBatchNorm2d(planes)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.bn2 = PrunableBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x

        # The input to a quantized conv can't come from a relu
        # It has to in [-1,1], because the output will be in [-1,1]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # Prune residual connections
            self.downsample._modules['0'].prune_or_zeroOut = self.conv1.prune_or_zeroOut
            self.downsample._modules['1'].prune_or_zeroOut = self.conv1.prune_or_zeroOut

            self.downsample._modules['0'].pruned_input = self.conv1.pruned_input
            self.downsample._modules['0'].pruned_output = self.conv2.pruned_output
            self.downsample._modules['1'].pruned = self.conv2.pruned_output
            residual = self.downsample(residual)
        else:
            if self.conv1.prune_or_zeroOut == 'prune':
                temp = torch.cuda.FloatTensor(residual.shape[0], self.conv1.pruned_input.shape[0],
                                              residual.shape[2], residual.shape[3]).fill_(0)
                temp[:, self.conv1.pruned_input != 1, :, :] = residual
                residual = temp[:, self.conv2.pruned_output != 1, :, :]

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = PrunableBatchNorm2d(planes)
        self.conv2 = Binaryconv3x3(planes, planes, stride=stride)
        self.bn2 = PrunableBatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = PrunableBatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # import pdb; pdb.set_trace()

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # Prune residual connections
            self.downsample._modules['0'].prune_or_zeroOut = self.conv1.prune_or_zeroOut
            self.downsample._modules['1'].prune_or_zeroOut = self.conv1.prune_or_zeroOut

            self.downsample._modules['0'].pruned_input = self.conv1.pruned_input
            self.downsample._modules['0'].pruned_output = self.conv3.pruned_output
            self.downsample._modules['1'].pruned = self.conv3.pruned_output
            residual = self.downsample(x)
        else:
            if self.conv1.prune_or_zeroOut == 'prune':
                temp = torch.cuda.FloatTensor(residual.shape[0], self.conv1.pruned_input.shape[0],
                                              residual.shape[2], residual.shape[3]).fill_(0)
                temp[:, self.conv1.pruned_input != 1, :, :] = residual
                residual = temp[:, self.conv3.pruned_output != 1, :, :]

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                PrunableBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        # The input to a quantized conv can't come from a relu
        # It has to in [-1,1], because the output will be in [-1,1]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)    # This relu might affect when quantizing activations of last layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)     # Required with binaryfc for gradients to be in about same range as weights

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000, block=Bottleneck, FLReal=False, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64

        # If first and last layer real
        if FLReal:
            self.conv1 = PrunableConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = PrunableBatchNorm2d(self.inplanes)
        else:
            self.conv1 = BinarizeConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = PrunableBatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # If first and last layer real
        if FLReal:
            self.fc = PrunableLinear(512 * block.expansion, num_classes)
            self.bn3 = lambda x: x
        else:
            self.fc = BinarizeLinear(512 * block.expansion, num_classes)
            self.bn3 = nn.BatchNorm1d(num_classes)

        # self.logsoftmax = nn.LogSoftmax(dim=1)      # Dimension 0 would be batch

        init_model(self)

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-2},
            20: {'lr': 5e-3},
            30: {'lr': 1e-3},
            40: {'lr': 5e-4},
            60: {'lr': 5e-5},
            70: {'lr': 5e-6},
            75: {'lr': 5e-7},
        }

        self.ReTrain = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            1: {'lr': 1e-3},
            2: {'lr': 5e-4},
            3: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, FLReal=False, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)

        # If first and last layer real
        if FLReal:
            self.conv1 = PrunableConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = PrunableBatchNorm2d(self.inplanes)
        else:
            self.conv1 = BinarizeConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = PrunableBatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # If first and last layer real
        if FLReal:
            self.fc = PrunableLinear(64 * self.inflate, num_classes)
            self.bn3 = lambda x: x
        else:
            self.fc = BinarizeLinear(64 * self.inflate, num_classes)
            self.bn3 = nn.BatchNorm1d(num_classes)

        # self.logsoftmax = nn.LogSoftmax(dim=1)      # Dimension 0 would be batch

        init_model(self)
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

        self.ReTrain = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            1: {'lr': 1e-3},
            2: {'lr': 5e-4},
            3: {'lr': 1e-4},
            4: {'lr': 1e-5}
        }


def resnet_binary(**kwargs):
    depth, dataset, FLReal = map(
        kwargs.get, ['depth', 'dataset', 'FLReal'])
    if dataset == 'tiny_imagenet' or dataset == 'imagenet':
        num_classes = 1000 if dataset == 'imagenet' else 200
        depth = depth or 18
        if depth == 18:
            return ResNet_imagenet(num_classes, BasicBlock, FLReal, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes, BasicBlock, FLReal, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes, Bottleneck, FLReal, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes, Bottleneck, FLReal, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes, Bottleneck, FLReal, layers=[3, 8, 36, 3])
    elif dataset == 'cifar10':
        num_classes = 10
        depth = depth or 18
        return ResNet_cifar10(num_classes, BasicBlock, FLReal, depth)
    elif dataset == 'cifar100':
        num_classes = 100
        depth = depth or 18
        return ResNet_cifar10(num_classes, BasicBlock, FLReal, depth)
