import torch
import torch.nn as nn
from typing import Optional


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
                 num_layers: Optional[int] = 1,
                 num_classes: Optional[int] = 10
                 ):
        super(BiGruModel, self).__init__()

        self.sen_rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.LayerNormal = LayerNormal(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x, _ = self.sen_rnn(x, None)
        x = self.LayerNormal(x)
        x = x[:, -1, :]
        x = self.fc(x)
        out_prob = torch.log_softmax(x, 1)
        return out_prob


def get_parameter_number(net, name):
    total_num = sum(p.numel() for p in net.parameters())
    return {'name: {}: ->:{}'.format(name, total_num)}


if __name__ == '__main__':
    gru_data = torch.rand(10, 16, 64)
    gru_model = BiGruModel()

    lm = gru_model(gru_data)
    p1 = get_parameter_number(gru_model, 'gru_model')
    print(lm.shape, p1)
