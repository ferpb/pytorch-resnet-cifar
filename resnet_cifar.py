import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn= fn

    def forward(self, x):
        return self.fn(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut_option="A"):
        super().__init__()

        assert stride == 1 or stride == 2
        assert out_planes == in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            if shortcut_option == 'A':
                self.downsample = LambdaLayer(lambda x:
                                              F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_planes//4, out_planes//4)))
            elif shortcut_option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(out_planes),
                )

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.downsample(identity)
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, shorcut_option="A"):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.stage1 = self._make_stage(block, 16, num_blocks, 1, shorcut_option)
        self.stage2 = self._make_stage(block, 32, num_blocks, 2, shorcut_option)
        self.stage3 = self._make_stage(block, 64, num_blocks, 2, shorcut_option)

        self.linear = nn.Linear(64, num_classes)

    def _make_stage(self, block_cls, planes, num_blocks, stride, shotcut_option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_cls(self.in_planes, planes, stride, shotcut_option))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet20():
    return ResNet(BasicBlock, 3)

def resnet32():
    return ResNet(BasicBlock, 5)

def resnet44():
    return ResNet(BasicBlock, 7)

def resnet56():
    return ResNet(BasicBlock, 9)

def resnet110():
    return ResNet(BasicBlock, 18)

def resnet1202():
    return ResNet(BasicBlock, 200)
