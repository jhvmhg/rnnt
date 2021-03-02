import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model import Transducer
from rnnt.optim import Optimizer
from rnnt.dataset import AudioDataset
from tensorboardX import SummaryWriter
from rnnt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer




def eval(config, model, validating_data, logger):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        preds = model.recognize(inputs, inputs_length)


        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]
        logger.info(preds)

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100


    val_loss = total_loss/(step+1)
    logger.info('-Validation-:, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (val_loss, cer))


    return cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    num_workers = config.training.num_gpu * 2
    train_dataset = AudioDataset(config.data, 'train')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=config.data.shuffle, num_workers=num_workers)
    logger.info('Load Train Set!')

    dev_dataset = AudioDataset(config.data, 'dev')
    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=False, num_workers=num_workers)
    logger.info('Load Dev Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    model = Transducer(config.model)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
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

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))

    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    if opt.mode == 'continue':
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




    _ = eval(config, model, validate_data, logger)




if __name__ == '__main__':
    main()
