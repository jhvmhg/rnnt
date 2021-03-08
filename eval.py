import os
import argparse
import yaml
import torch
import torch.utils.data

from data.dataset import AudioDataLoader, Batch_RandomSampler
from utils.utils import AttrDict, init_logger, computer_cer
from utils.checkpoint import new_model
from ctc.ctc_decoder import BeamCTCDecoder


def eval(config, model, validating_data, logger, visualizer=None, beamctc_decoder=None):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        preds = model.recognize(inputs, inputs_length)
        if beamctc_decoder:
            results_strings, results_tensor, scores, offsets, seq_lens = beamctc_decoder.decode(inputs, inputs_length)
            preds = [i[0] for i in results_tensor]
        else:
            preds = model.recognize(inputs, inputs_length)
        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-:(%.5f%%), CER: %.5f %%' % (process, cer))
            logger.info('preds:' + validating_data.dataset.decode(preds[0]))
            logger.info('transcripts:' + validating_data.dataset.decode(transcripts[0]))
            logger.info('cer_num:' + str(dist))

    val_loss = total_loss / (step + 1)
    logger.info('-Validation:, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer)

    return cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='eval.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.data.name, 'exp', config.model.type, config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    num_workers = 6 * (config.training.num_gpu if config.training.num_gpu > 0 else 1)
    batch_size = config.data.batch_size * config.training.num_gpu if config.training.num_gpu > 0 else config.data.batch_size

    dev_dataset = (config.data, 'dev')
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

    if config.evaling.num_gpu == 0:
        checkpoint = torch.load(config.evaling.load_model, map_location='cpu')
    else:
        checkpoint = torch.load(config.evaling.load_model)
    logger.info(str(checkpoint.keys()))

    model = new_model(config, checkpoint)

    beamctc_decoder = None
    if config.model.type == "ctc" and config.evaling.lm_model:
        beamctc_decoder = BeamCTCDecoder(config.data.vocab, model, lm_path=config.evaling.lm_model,
                                         log_probs_input=True, beam_width=10)
    if config.training.num_gpu > 0:
        model = model.cuda()

    _ = eval(config, model, validate_data, logger, beamctc_decoder)


if __name__ == '__main__':
    main()
