import paddle
import paddle.nn as nn
import paddle.nn.functional  as F

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.initializer.XavierUniform(m.weight, gain=np.sqrt(2))
        nn.initializer.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.initializer.constant(m.weight, 1)
        nn.initializer.constant(m.bias, 0)

class wide_basic(nn.Layer):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Layer):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2D(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = paddle.flatten(out, 1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net = Wide_ResNet(depth=28, widen_factor=20, dropout_rate=0.3,num_classes=10)
    # print(net)
    FLOPs = paddle.flops(net, [1, 3, 32, 32], print_detail=False)
    print(FLOPs)
