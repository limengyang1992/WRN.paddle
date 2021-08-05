
import math
import paddle
import paddle.nn as nn
import paddle.fluid  as F




class BasicBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.3):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D (in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2D(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1 )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.layers.dropout(out, dropout_prob=self.droprate)
        out = self.conv2(out)
        return paddle.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Layer):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.3):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Layer):
    def __init__(self, depth, widen_factor=1, dropout_rate=0.3,num_classes=10):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2D(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2D(nChannels[3])
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.feature_num = nChannels[3]
        self.fc = nn.Linear(self.feature_num, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = paddle.nn.functional.avg_pool2d(out, 8)
        # out = out.view(-1, self.nChannels)
        out = paddle.flatten(out, 1)
        out = self.fc(out)
        return out #, self.fc(out)


if __name__ == '__main__':
    net = WideResNet(depth=28, widen_factor=20, dropout_rate=0.3,num_classes=10)
    # print(net)
    FLOPs = paddle.flops(net, [1, 3, 32, 32], print_detail=False)
    print(FLOPs)

