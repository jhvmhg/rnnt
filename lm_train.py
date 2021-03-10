import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model import LM, Transducer
from ctc.model import CTC
from utils.optim import Optimizer
from data.dataset import LmDataset, AudioDataLoader, Batch_RandomSampler
from tensorboardX import SummaryWriter
from utils.utils import AttrDict, init_logger
from utils.checkpoint import save_model, load_model, load_ctc_model, load_language_model

from train import train, eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.data.name, 'exp', config.model.type, config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.training.gpus

    config.training.num_gpu = len(config.training.gpus.split(","))
    num_workers = 6 * (config.training.num_gpu if config.training.num_gpu > 0 else 1)
    batch_size = config.data.batch_size * config.training.num_gpu if config.training.num_gpu > 0 else config.data.batch_size

    train_dataset = LmDataset(config.data, 'train')
    train_sampler = Batch_RandomSampler(len(train_dataset),
                                        batch_size=batch_size, shuffle=config.data.shuffle)
    training_data = AudioDataLoader(
        dataset=train_dataset,
        num_workers=num_workers,
        batch_sampler=train_sampler
    )
    logger.info('Load Train Set!')

    dev_dataset = LmDataset(config.data, 'dev')
    dev_sampler = Batch_RandomSampler(len(dev_dataset),
                                      batch_size=batch_size, shuffle=config.data.shuffle)
    validate_data = AudioDataLoader(
        dataset=dev_dataset,
        num_workers=num_workers,
        batch_sampler=dev_sampler
    )
    logger.info('Load Dev Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    if config.model.type == "transducer":
        model = Transducer(config.model)
    elif config.model.type == "ctc":
        model = CTC(config.model)
    elif config.model.type == "lm":
        model = LM(config.model)
    else:
        raise NotImplementedError

    if config.training.load_model:
        if config.training.num_gpu == 0:
            checkpoint = torch.load(config.training.load_model, map_location='cpu')
        else:
            checkpoint = torch.load(config.training.load_model)
        logger.info(str(checkpoint.keys()))
        load_model(model, checkpoint)
        logger.info('Loaded model from %s' % config.training.new_model)

    if config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    # n_params, enc, dec = count_parameters(model)
    # logger.info('# the number of parameters in the whole model: %d' % n_params)
    # logger.info('# the number of parameters in the Encoder: %d' % enc)
    # logger.info('# the number of parameters in the Decoder: %d' % dec)
    # logger.info('# the number of parameters in the JointNet: %d' %
    #             (n_params - dec - enc))

    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    if opt.mode == 'continue':
        if not config.training.load_model:
            raise Exception("if mode is 'continue', need 'config.training.load_model'")
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    logger.info(model)
    for epoch in range(start_epoch, config.training.epochs):

        train(epoch, config, model, training_data,
              optimizer, logger, visualizer)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)


        if config.training.eval_or_not:
            _ = eval(epoch, config, model, validate_data, logger, visualizer)

        if epoch >= config.optim.begin_to_adjust_lr:
            optimizer.decay_lr()
            # early stop
            if optimizer.lr < 1e-6:
                logger.info('The learning rate is too low to train.')
                break
            logger.info('Epoch %d update learning rate: %.6f' %
                        (epoch, optimizer.lr))

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
