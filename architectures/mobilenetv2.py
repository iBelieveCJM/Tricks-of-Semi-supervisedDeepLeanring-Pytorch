#!coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride==1 and in_planes!=out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class _MobileNetV2(nn.Module):
    
    def __init__(self, cfg, num_classes):
        super(_MobileNetV2, self).__init__()
        self.cfg = cfg
        # note. change conv1 stride 2->1 for cifar10
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc1 = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # note. change pooling kernel_size 7->4 for cifar10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc1(out)

def MobileNetV2(num_classes):
    #(expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 1),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6,160, 3, 2),
        (6,320, 1, 1)
    ]
    return _MobileNetV2(cfg, num_classes)

def test():
    print('--- run mobilenet test ---')
    x = torch.randn(2,3,32,32)
    net = MobileNetV2(10)
    y = net(x)
    print(y.size())
