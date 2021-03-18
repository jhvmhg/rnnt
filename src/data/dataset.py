import copy
import os
import torch
from torch.utils.data import Sampler, DataLoader
from src.data.utils import *


class myDataset:
    def __init__(self, config, dataset_type):

        self.type = dataset_type
        self.name = config.name
        self.vocab = config.vocab

        # if self.config.encoding:
        self.unit2idx = get_dict_from_scp(self.vocab, int)  # same function as get self.utt2num_frames_dict
        self.idx2unit = dict([(i, c) for (i, c) in enumerate(self.unit2idx)])

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length

    def __len__(self):
        raise NotImplementedError

    def get_targets_dict(self, text):
        targets_dict = {}
        with codecs.open(text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                contents = parts[1:]

                labels = self.encode(contents)

                targets_dict[utt_id] = labels
        return targets_dict

    def encode(self, seq):

        return [self.unit2idx[unit] if unit in self.unit2idx else self.unit2idx['<unk>'] for unit in seq]

    def decode(self, seq, rm_blk=False):

        if rm_blk:
            return " ".join([self.idx2unit[int(i)] for i in seq if i > 0])
        return " ".join([self.idx2unit[int(i)] for i in seq])


class AudioDataset(myDataset):
    def __init__(self, config, dataset_type):
        super(AudioDataset, self).__init__(config, dataset_type)  # dataset_type :"train", "dev", "test"

        self.config = config
        self.text = os.path.join(config.__getattr__(dataset_type), 'text')
        self.targets_dict = self.get_targets_dict(self.text)
        self.utt2num_frames_txt = os.path.join(config.__getattr__(dataset_type), 'utt2num_frames')

        self.short_first = config.short_first

        self.arkscp = os.path.join(config.__getattr__(dataset_type), 'feats.scp')

        self.left_context_width = config.left_context_width if config.left_context_width else 0
        self.right_context_width = config.right_context_width if config.right_context_width else 0
        self.frame_rate = config.frame_rate if config.frame_rate else 10
        self.apply_cmvn = config.apply_cmvn if config.apply_cmvn else False

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.__getattr__(dataset_type), 'utt2spk'), 'r') as fid:
                for line in fid:
                    parts = line.strip().split()
                    self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.__getattr__(dataset_type), 'cmvn.scp')
            self.cmvn_stats_dict = get_cmvn_dict(self.cmvnscp)

        self.feats_list, self.feats_dict = get_feats_list(self.arkscp)
        self.utt2num_frames_dict = get_dict_from_scp(self.utt2num_frames_txt, lambda x: int(x))

        if self.short_first:
            self.sorted_list = sorted(self.utt2num_frames_dict.items(), key=lambda x: x[1], reverse=False)
        else:
            self.sorted_list = None

        self.check_speech_and_text()
        self.lengths = len(self.feats_list)

    def __getitem__(self, index):
        if self.sorted_list is not None:
            utt_id = self.sorted_list[index][0]
        else:
            utt_id = self.feats_list[index]

        feat_scp = self.feats_dict[utt_id]
        seq = self.targets_dict[utt_id]

        target = np.array(seq)
        feature = kaldiio.load_mat(feat_scp)

        if self.apply_cmvn:
            spk_id = self.utt2spk[utt_id]
            stats = self.cmvn_stats_dict[spk_id]
            feature = cmvn(feature, stats)

        feature = concat_frame(feature, self.left_context_width, self.right_context_width)
        feature = subsampling(feature, self.frame_rate)

        # if features 长度 > max_input_length,只保留前面部分
        if feature.shape[0] >= self.config.max_input_length:
            feature = feature[:self.config.max_input_length, ]
        if target.shape[0] >= self.max_target_length:
            target = target[:self.max_target_length]

        inputs_length = np.array(feature.shape[0]).astype(np.int64)
        targets_length = np.array(target.shape[0]).astype(np.int64)

        return feature, inputs_length, target, targets_length

    def __len__(self):
        return self.lengths

    def check_speech_and_text(self):
        featslist = copy.deepcopy(self.feats_list)
        for utt_id in featslist:
            if utt_id not in self.targets_dict:
                self.feats_list.remove(utt_id)


class LmDataset(myDataset):

    def __init__(self, config, dataset_type):
        super(LmDataset, self).__init__(config, dataset_type)
        self.text = config.__getattr__(dataset_type)
        self.targets_dict = self.get_targets_dict(self.text)
        self.max_target_length = config.max_target_length
        self.short_first = config.short_first
        if self.short_first:
            self.sorted_list = sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)
        else:
            self.sorted_list = list(self.targets_dict)

        self.lengths = len(self.sorted_list)

    def __getitem__(self, index):
        utt_id = self.sorted_list[index][0]

        seq = self.targets_dict[utt_id]
        seq_ids = np.array(seq)

        if seq_ids.shape[0] >= self.max_target_length:
            seq_ids = seq_ids[:self.max_target_length]

        # input = seq_ids[:-1]
        # targets = seq_ids[1:]
        input = np.concatenate((np.array([0]), seq_ids[:-1]))
        targets = seq_ids
        if input.shape[0] == 0:
            input = np.array([0])
            targets = np.array([0])
        inputs_length = np.array(input.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        return input, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    features = [b[0] for b in batch]
    inputs_length = np.array([b[1] for b in batch])
    targets = [b[2] for b in batch]
    targets_length = np.array([b[3] for b in batch])

    max_inputs_length = max(inputs_length)
    max_targets_length = max(targets_length)

    features = pad_np(features, max_inputs_length)
    targets = pad_np(targets, max_targets_length)

    return torch.tensor(features), torch.tensor(inputs_length), torch.tensor(targets), torch.tensor(targets_length)


class Batch_RandomSampler(Sampler):
    r"""
    iter get [6, 7, 8]
             [0, 1, 2]
             [3, 4, 5]
             [9]
    """

    def __init__(self, index_length, batch_size, shuffle=True):
        self.index_length = index_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.len = (self.index_length + self.batch_size - 1) // self.batch_size
        self.index = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            index_list = torch.randperm(self.len, generator=g).tolist()
        else:
            index_list = [i for i in range(self.len)]

        while self.index < self.len:
            k = index_list[self.index]  # 第k个group
            start = k * self.batch_size
            end = start + self.batch_size
            end = end if end < self.index_length else self.index_length

            yield [i for i in range(end - 1, start - 1, -1)]
            self.index += 1

        self.index = 0
        self.epoch += 1

    def __len__(self):
        return self.len

    def set_epoch(self, epoch):
        self.epoch = epoch
