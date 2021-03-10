import torch
import logging
import editdistance

from shutil import move


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def computer_cer(preds, labels):
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total


def get_saved_folder_name(config):
    return '_'.join([config.data.name, config.training.save_model])


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)
        else:
            pass


def add_space(path, old_path):
    with open(path, "r") as f_in:
        lines = f_in.readlines()

    move(path, old_path)

    with open(path, "w") as f_out:
        for line in lines:
            utt, txt = line.split()
            f_out.write(utt + " " + " ".join(list(txt)) + "\n")


import matplotlib.pyplot as plt
from pylab import *

zhfont1 = matplotlib.font_manager.FontProperties(
    fname="/home1/meichaoyang/workspace/git/kws_ctc_no2/data/SourceHanSansSC-Bold.otf")


def show_ctc_loss(utt_prob, target, idx2unit, save_path):
    plt.plot(utt_prob[:, 0].detach().numpy(), linewidth=0.1)
    for j in target:
        if j != 0:
            plt.plot(utt_prob[:, int(j)].detach().numpy(), linewidth=0.5, linestyle="-.", label=idx2unit[int(j)])

    legend(loc='upper right', prop=zhfont1)
    savefig(save_path, dpi=440)
