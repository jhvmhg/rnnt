import torch
from rnnt.model import Transducer
from ctc.model import CTC
from rnnt.encoder import BaseEncoder, CNN_LSTM
from ctc.deep_speech import DeepSpeech
from rnnt.decoder import BaseDecoder


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
            bidirectional=config.enc.bidirectional
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


def save_model(model, optimizer, config, save_name):
    if model.config.type == "transducer":
        save_rnn_t_model(model, optimizer, config, save_name)
    elif model.config.type == "ctc":
        save_ctc_model(model, optimizer, config, save_name)
    else:
        raise NotImplementedError


def new_model(config, checkpoint):
    if config.model.type == "transducer":
        model = Transducer(config.model)
        load_rnn_t_model(model, checkpoint)
    elif config.model.type == "ctc":
        model = CTC(config.model)
        load_ctc_model(model, checkpoint)
    else:
        raise NotImplementedError

    return model


def save_rnn_t_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def load_rnn_t_model(model, checkpoint):
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])


def save_ctc_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def load_ctc_model(model, checkpoint):
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.project_layer.load_state_dict(checkpoint['project_layer'])


def save_language_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def load_language_model(model, checkpoint):
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.project_layer.load_state_dict(checkpoint['project_layer'])
