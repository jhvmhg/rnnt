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
from data.dataset import LmDataset, LMDataLoader, Batch_RandomSampler
from tensorboardX import SummaryWriter
from utils.utils import AttrDict, init_logger, count_parameters, computer_cer
from utils.checkpoint import save_model, load_rnn_t_model, load_ctc_model


def train(epoch, config, model, training_data, optimizer, logger, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        # max_inputs_length = inputs_length.max().item()
        # max_targets_length = targets_length.max().item()
        # inputs = inputs[:, :max_inputs_length, :]
        # targets = targets[:, :max_targets_length]

        if config.optim.step_wise_update:
            optimizer.step_decay_lr()

        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        total_loss += loss.item()

        if config.training.max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm)
        else:
            grad_norm = 0

        optimizer.step()

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end - start))

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step + 1), end_epoch - start_epoch))




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

    num_workers = 6 * (config.training.num_gpu if config.training.num_gpu > 0 else 1)
    batch_size = config.data.batch_size * config.training.num_gpu if config.training.num_gpu > 0 else config.data.batch_size

    train_dataset = LmDataset(config.data, 'train')
    train_sampler = Batch_RandomSampler(len(train_dataset),
                                        batch_size=batch_size, shuffle=config.data.shuffle)
    training_data = LMDataLoader(
        dataset=train_dataset,
        num_workers=num_workers,
        batch_sampler=train_sampler
    )
    logger.info('Load Train Set!')

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
        print(str(checkpoint.keys()))
        if config.model.type == "transducer":
            load_rnn_t_model(model, checkpoint)
        elif config.model.type == "ctc":
            load_ctc_model(model, checkpoint)
        else:
            raise NotImplementedError
        logger.info('Loaded model from %s' % config.training.new_model)
    elif config.training.load_encoder or config.training.load_decoder:
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