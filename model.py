import torch
import torch.nn as nn
from loss import info_nce_loss

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.w = nn.Linear(input_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x):
        x = self.w(x)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output

class Net(nn.Module):
    def __init__(self, input_size_left, input_size_right, num_heads, hidden_size, criterion, device):
        super(Net, self).__init__()
        self.input_size_left = input_size_left
        self.input_size_right = input_size_right
        self.layers_left = nn.ModuleList([nn.Linear(input_size, hidden_size) for input_size in input_size_left])
        self.layers_right = nn.ModuleList([SelfAttention(input_size[0], hidden_size, input_size[1]) if isinstance(input_size, list) else nn.Linear(input_size, hidden_size) for input_size in input_size_right])
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.criterion = criterion
        self.device = device

    def inference(self, left, right):
        _left = [layer(x.to(self.device)) for layer, x in zip(self.layers_left, left)]
        _right = [layer(x.to(self.device)) for layer, x in zip(self.layers_right, right)]
        _left = torch.cat(_left, dim = 1)
        _right = torch.cat(_right, dim = 1)
        _left = _left.permute(1, 0, 2)
        _right = _right.permute(1, 0, 2)
        attn_output, attn_weight = self.multihead_attention(_left, _right, _right)
        return attn_output, attn_weight

    def forward(self, left_1, right_1, left_2, right_2):
        y1, _ = self.inference(left_1, right_1)
        y2, _ = self.inference(left_2, right_2)
        y1 = y1.squeeze()
        y2 = y2.squeeze()
        z = torch.cat([y1, y2], dim = 0)
        logits, labels = info_nce_loss(z, len(z) // 2, 2)
        loss = self.criterion(logits, labels)
        return loss