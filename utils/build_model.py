from rnnt.encoder import BaseEncoder, CNN_LSTM
from rnnt.decoder import BaseDecoder
from ctc.deep_speech import DeepSpeech

"""
统一在这里构建细分模型，主要是编码器和解码器
"""

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
            cnn1_ksize=tuple([int(i) for i in config.enc.cnn1_ksize.split(",")]),
            cnn1_stride=tuple([int(i) for i in config.enc.k1_stride.split(",")]),
            cnn2_ksize=tuple([int(i) for i in config.enc.cnn2_ksize.split(",")]),
            cnn2_stride=tuple([int(i) for i in config.enc.k2_stride.split(",")]),
            bidirectional=config.enc.bidirectional,
            input_sorted=config.enc.input_sorted
        )
    else:
        raise NotImplementedError


def build_decoder(config):
    if config.dec.type == 'lstm':
        return BaseDecoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.dec.output_size,
            n_layers=config.dec.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight
        )
    else:
        raise NotImplementedError
