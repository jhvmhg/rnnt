import torch
from src.rnnt import Transducer, LM
from src.ctc import CTC


def save_model(model, optimizer, config, save_name):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if model.config.type == "transducer":
        save_rnn_t_model(model, optimizer, config, save_name)
    elif model.config.type == "ctc":
        save_ctc_model(model, optimizer, config, save_name)
    elif model.config.type == "lm":
        save_language_model(model, optimizer, config, save_name)
    else:
        raise NotImplementedError

def load_model(model, checkpoint):
    if model.config.type == "transducer":
        load_rnn_t_model(model, checkpoint)
    elif model.config.type == "ctc":
        load_ctc_model(model, checkpoint)
    elif model.config.type == "lm":
        load_language_model(model, checkpoint)
    else:
        raise NotImplementedError


def new_model(config, checkpoint):
    if config.model.type == "transducer":
        model = Transducer(config.model)
        load_rnn_t_model(model, checkpoint)
    elif config.model.type == "ctc":
        model = CTC(config.model)
        load_ctc_model(model, checkpoint)
    elif config.model.type == "lm":
        model = LM(config.model)
        load_language_model(model, checkpoint)
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
