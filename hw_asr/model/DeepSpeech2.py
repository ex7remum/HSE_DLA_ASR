from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel


class BatchNormRNN(nn.Module):
    def __init__(self, input_size, hidden_size, is_bidirectional, use_batch_norm, dropout):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, bias=False,
                          batch_first=True, dropout=dropout,
                          bidirectional=is_bidirectional)
        self.batch_norm = use_batch_norm
        self.is_bidirectional = is_bidirectional
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):
        if self.batch_norm:
            bs, length, dim = x.size()
            x = x.view(-1, dim)
            x = self.bn(x)
            x = x.view(bs, length, dim)
        x, _ = self.rnn(x)
        if self.is_bidirectional:
            batch_size, length, _ = x.size()
            x = x.view(batch_size, length, 2, -1).sum(dim=2).view(batch_size, length, -1)
        return x


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, n_layers_rnn, hidden_size, is_bidirectional, dropout, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(41, 11),
                      stride=(2, 2),
                      padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(21, 11),
                      stride=(2, 1),
                      padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        assert n_feats % 8 == 0
        self.input_size = n_feats * 8

        is_bidirectional = bool(is_bidirectional)  # otherwise not working with config
        self.first_rnn = BatchNormRNN(input_size=self.input_size,
                                      hidden_size=hidden_size,
                                      is_bidirectional=is_bidirectional,
                                      use_batch_norm=False,
                                      dropout=dropout)

        rnn_layers = []

        for i in range(n_layers_rnn - 1):
            rnn_layers.append(BatchNormRNN(input_size=hidden_size,
                                           hidden_size=hidden_size,
                                           is_bidirectional=is_bidirectional,
                                           use_batch_norm=True,
                                           dropout=dropout))
        self.rnn_layers = nn.Sequential(*rnn_layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.unsqueeze(1)  # (bs, 1, length, time)
        spectrogram = self.conv(spectrogram)  # (bs, channels, length, time)
        spectrogram = spectrogram.permute(0, 3, 1, 2)  # (bs, time, channels, length)
        batch_size, time, n_channels, length = spectrogram.size()
        spectrogram = spectrogram.view(batch_size, time, -1).contiguous()  # (bs, time, input)
        spectrogram = self.first_rnn(spectrogram)
        spectrogram = self.rnn_layers(spectrogram)
        spectrogram = self.out(spectrogram)
        return {"logits": spectrogram}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
