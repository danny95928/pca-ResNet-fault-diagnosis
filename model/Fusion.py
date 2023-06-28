import torch
import torch.nn as nn
import math
from typing import Optional
import numpy as np
import torch.nn.functional as F


class LayerNormal(nn.Module):
    def __init__(self, hidden_size, esp=1e-6):
        super(LayerNormal, self).__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = torch.mean(input=x, dim=-1, keepdim=True)
        sigma = torch.std(input=x, dim=-1, keepdim=True).clamp(min=self.esp)
        out = (x - mu) / sigma
        out = out * self.weight.expand_as(out) + self.bias.expand_as(out)
        return out


class BiGruModel(nn.Module):
    def __init__(self,
                 input_size: Optional[int] = 64,
                 hidden_size: Optional[int] = 256,
                 num_layers: Optional[int] = 1
                 ):
        super(BiGruModel, self).__init__()

        self.sen_rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.LayerNormal = LayerNormal(hidden_size)

    def forward(self, x):
        x, _ = self.sen_rnn(x, None)
        x = self.LayerNormal(x)
        x = x[:, -1, :]
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(BasicBlock, 128, [[1, 1], [1, 1]])
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv4 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv5 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])

    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out5 = out5.reshape(out5.size(0), -1)
        return out5


class Fusion(torch.nn.Module):
    def __init__(self, num_classes):
        super(Fusion, self).__init__()

        self.fc = nn.Linear(384, num_classes)
        self._init_weights()

        self.resnet = ResNet()
        self.biGru = BiGruModel()

    def forward(self, rnn_array, img_array):
        rnn_array = self.biGru(rnn_array)
        img_array = self.resnet(img_array)

        x = torch.cat((torch.tanh(rnn_array), torch.tanh(img_array)), dim=-1)
        x = torch.log_softmax(self.fc(x), dim=1)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_parameter_number(net, name):
    total_num = sum(p.numel() for p in net.parameters())
    return {'name{}: ->:{}'.format(name, total_num)}


if __name__ == '__main__':
    model = Fusion(num_classes=10)
    model.eval()
    batch = 10
    img_arrays = torch.randn(batch, 1, 32, 32)
    rnn_arrays = torch.randn(batch, 16, 64)
    p = get_parameter_number(model, 'fusion_model')
    y = model(rnn_arrays, img_arrays)
    print(y.size(), p)

