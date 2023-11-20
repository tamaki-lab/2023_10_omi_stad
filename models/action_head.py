import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    def __init__(self, in_d: int=256, nhead: int=8, n_layers: int=2, n_classes: int=24):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=in_d, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(256, n_classes + 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    input = torch.rand(10, 256)
    head = ActionHead()
    output = head(input)
    print(output.shape)
