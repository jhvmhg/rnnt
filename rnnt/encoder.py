import torch
import torch.nn as nn
from rnnt.deep_speech import *


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, kernal_size, pad, rnn_input_size, rnn_hidden_size, output_size, n_layers,
                 dropout=0.2, bidirectional=True):
        super(CNN_LSTM, self).__init__()

        self.conv1d = nn.Conv1d(input_size, rnn_input_size, kernal_size, padding=pad)
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * rnn_hidden_size if bidirectional else rnn_hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden


def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    elif config.enc.type == 'cnn_lstm':
        return CNN_LSTM(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    elif config.enc.type == 'deep_speech':
        return DeepSpeech(
            input_size=config.feature_dim,
            rnn_hidden_size=config.enc.hidden_size,
            rnn_hidden_layers=config.enc.n_layers,
            output_size=config.enc.output_size,
            cnn1_ksize=tuple([int(i) for i in config.model.enc.k1_size.split(",")]),
            cnn2_ksize=tuple([int(i) for i in config.model.enc.k2_size.split(",")]),
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError
