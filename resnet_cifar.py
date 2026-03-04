import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn = fn

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
            if shortcut_option == "A":
                self.downsample = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_planes // 4, out_planes // 4))
                )
            elif shortcut_option == "B":
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

        self.apply(self._init_weights)

    def _make_stage(self, block_cls, planes, num_blocks, stride, shotcut_option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_cls(self.in_planes, planes, stride, shotcut_option))
            self.in_planes = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet20(num_classes=10):
    return ResNet(BasicBlock, 3, num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, 5, num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, 7, num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, 9, num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, 18, num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, 200, num_classes)
