# ResNet for CIFAR datasets

PyTorch implementation of CIFAR ResNets (`resnet20` to `resnet1202`) with a simple training script.

## Train

```bash
python train.py --config resnet20 --device cuda --log
```

## References

- Paper: https://arxiv.org/abs/1512.03385
- Original paper code repo: https://github.com/KaimingHe/deep-residual-networks
- https://github.com/akamaster/pytorch_resnet_cifar10
