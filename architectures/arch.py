#!coding:utf-8
from functools import wraps

from architectures.lenet import LeNet
from architectures.vgg import VGG11, VGG13, VGG16, VGG19
from architectures.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from architectures.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from architectures.densenet import DenseNet_cifar, DenseNet121, DenseNet169, DenseNet201, DenseNet161
from architectures.resnext import ResNeXt29_2x64d,ResNeXt29_4x64d,ResNeXt29_8x64d,ResNeXt29_32x4d
from architectures.senet import SENet18
from architectures.dpn import DPN26, DPN92
from architectures.shufflenet import ShuffleNetG2, ShuffleNetG3
from architectures.mobilenet import MobileNetV1
from architectures.mobilenetv2 import MobileNetV2
from architectures.convlarge import convLarge

arch = {
        'lenet': LeNet,
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'preact_resnet18': PreActResNet18,
        'preact_resnet34': PreActResNet34,
        'preact_resnet50': PreActResNet50,
        'preact_resnet101': PreActResNet101,
        'preact_resnet152': PreActResNet152,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'densenet161': DenseNet161,
        'densenet': DenseNet_cifar,
        'resnext29_2x64d': ResNeXt29_2x64d,
        'resnext29_4x64d': ResNeXt29_4x64d,
        'resnext29_8x64d': ResNeXt29_8x64d,
        'resnext29_32x4d': ResNeXt29_32x4d,
        'senet': SENet18,
        'dpn26': DPN26,
        'dpn92': DPN92,
        'shuffleG2': ShuffleNetG2,
        'shuffleG3': ShuffleNetG3,
        'mobileV1': MobileNetV1,
        'mobileV2': MobileNetV2,
        'cnn13': convLarge
        }


def RegisterArch(arch_name):
    """Register a model
    you must import the file where using this decorator
    for register the model function
    """
    def warpper(f):
        arch[arch_name] = f
        return f
    return warpper
