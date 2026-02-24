from resnet_cifar import *


def test_num_params():
    model = resnet20()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 2) == 0.27

    model = resnet32()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 2) == 0.46

    model = resnet44()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 2) == 0.66

    model = resnet56()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 2) == 0.85

    model = resnet110()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 1) == 1.7

    model = resnet1202()
    num_params = sum(p.numel() for p in model.parameters())
    assert round(num_params / 1e6, 1) == 19.4


def test_shortcut_options():
    model_a = ResNet(BasicBlock, 3, num_classes=10, shorcut_option="A")
    model_b = ResNet(BasicBlock, 3, num_classes=10, shorcut_option="B")
    input = torch.randn(128, 3, 32, 32)
    assert model_a(input).size() == model_b(input).size()
