import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in, seq_len):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.splconv = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3, padding=padding)
        self.splnorm = nn.BatchNorm1d(seq_len)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.downConv2 = nn.Conv1d(in_channels=2 * c_in, out_channels=c_in, kernel_size=1)

    def forward(self, x):
        x_1 = self.downConv(x.permute(0, 2, 1))
        x_1 = self.norm(x_1)
        x_1 = self.activation(x_1)

        #x = self.splconv(x_1.permute(0, 2, 1)).permute(0, 2, 1)

        # ---add---
        x_2 = self.splconv(x)
        x_2 = self.splnorm(x_2)
        x_2 = self.activation(x_2).permute(0, 2, 1)

        x = torch.cat([x_1, x_2], dim=1)
        x = self.downConv2(x)
        #------

        x = self.maxPool(x_1)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 矩陣翻轉做計算
        self.gru = nn.LSTM(input_size=d_model, hidden_size=d_model)
        #self.gru = nn.GRU(input_size=d_model, hidden_size=d_model // 2, bidirectional=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, prev=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn, prev = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            prev=prev
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y, _ = self.gru(y)
        return self.norm2(x + y), attn, prev


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            prev = None
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn, prev = attn_layer(x, attn_mask=attn_mask, prev=prev)
                x = conv_layer(x)
                attns.append(attn)
            x, attn, prev = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns
