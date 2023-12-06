import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchinfo


class ActionHead(nn.Module):
    def __init__(self, in_d: int=256, n_layers: int=2, n_classes: int=24, pos_ecd: tuple=(True, "cat", 32)):
        super().__init__()
        self.use_pos_ecd = pos_ecd[0]
        if pos_ecd[0]:
            if pos_ecd[1] == "cat":
                self.pos_encoder = PositionalEncoding(pos_ecd[2], ecd_type=pos_ecd[1])
                dim = in_d + pos_ecd[2]
            elif pos_ecd[1] == "add":
                self.pos_encoder = PositionalEncoding(in_d, ecd_type=pos_ecd[1])
                dim = in_d
        else:
            dim = in_d
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=in_d)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(dim, n_classes + 1)

    def forward(self, x, frame_indices=None):
        if self.use_pos_ecd:
            x = self.pos_encoder(x, frame_indices)
        x = self.transformer_encoder(x)
        x = self.head(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Imported from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Changed to assume online (batch size=1) instead of batch
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, ecd_type: str = "cat", cycle: float = 2500.0):
    # def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, ecd_type: str = "cat", cycle: float = 10000.0):
        super().__init__()
        self.ecd_type = ecd_type

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(cycle) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, frame_indices: list[int] = None) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, embedding_dim]``
        """
        if self.ecd_type == "cat":
            if frame_indices:
                x = torch.cat((x, self.pe[torch.Tensor(frame_indices).to(torch.int64), :]), dim=1)
            else:
                x = torch.cat((x, self.pe[:x.size(0)]), dim=1)
        elif self.ecd_type == "add":
            if frame_indices:
                x = x + self.pe[torch.Tensor(frame_indices).to(torch.int64), :]
            else:
                x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.mlp = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.reshape(-1, 2048)
        x = self.mlp(self.dropout(x))
        return x


class Extractor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2048, )
        self.dropout = nn.Dropout(p=0.1)
        self.mlp = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.reshape(-1, 2048)
        x = self.mlp(self.dropout(x))
        return x


class ActionHead2(nn.Module):
    def __init__(self, in_d: int=256, n_layers: int=2, n_classes: int=24, pos_ecd: tuple=(True, "cat", 32)):
        super().__init__()
        self.extractor = Extractor()  # (n,2048,32,32) -> (n,256)
        self.use_pos_ecd = pos_ecd[0]
        if pos_ecd[0]:
            if pos_ecd[1] == "cat":
                self.pos_encoder = PositionalEncoding(pos_ecd[2], ecd_type=pos_ecd[1])
                dim = in_d + pos_ecd[2]
            elif pos_ecd[1] == "add":
                self.pos_encoder = PositionalEncoding(in_d, ecd_type=pos_ecd[1])
                dim = in_d
        else:
            dim = in_d
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=in_d)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.head = nn.Linear(dim, n_classes + 1)

    def forward(self, tgt, memory, frame_indices=None):
        tgt = self.extractor(tgt)
        if self.use_pos_ecd:
            tgt = self.pos_encoder(tgt, frame_indices)
            memory = self.pos_encoder(memory, frame_indices)
        x = self.transformer_decoder(tgt, memory)
        x = self.head(x)
        return x

class X3D_XS(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load('facebookresearch/pytorchvideo', "x3d_xs", pretrained=True)
        self.blocks = nn.ModuleList([b for b in model.blocks[:-1]])
        self.blocks.append(model.blocks[5].pool)
        self.avgpool = nn.AdaptiveAvgPool3d(1)


    def forward(self, x):
        for f in self.blocks:
            x = f(x)
        x = x.reshape(-1, 2048, 2, 2)
        # x = self.avgpool(x).reshape(-1, 2048)
        return x

if __name__ == "__main__":
    model = X3D_XS()
    # model = torch.hub.load('facebookresearch/pytorchvideo', "x3d_xs", pretrained=True)

    # head = ActionHead2()
    # input = torch.rand([2, 3, 4, 182, 182])
    # print(model(input).shape)
    # exit()

    torchinfo.summary(
        model=model,
        # model=head,
        # input_size=((10, 2048, 32, 32), (10, 256)),
        # input_size=((10, 256+32), (16, 256)),
        input_size=(2, 3, 4, 182, 182),
        depth=3,
        col_names=["input_size",
                   "output_size"],
        row_settings=("var_names",))
