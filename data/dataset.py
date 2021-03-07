import codecs
import copy
import numpy as np
import os
import kaldiio
import torch
from torch.utils.data import Sampler, DataLoader
from data.utils import pad_np, get_dict_from_scp, cmvn


class myDataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.frame_rate = config.frame_rate
        self.apply_cmvn = config.apply_cmvn

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.vocab = config.vocab

        self.arkscp = os.path.join(config.__getattr__(type), 'feats.scp')

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.__getattr__(type), 'utt2spk'), 'r') as fid:
                for line in fid:
                    parts = line.strip().split()
                    self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.__getattr__(type), 'cmvn.scp')
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

        self.feats_list, self.feats_dict = self.get_feats_list()

    def __len__(self):
        raise NotImplementedError

    def get_feats_list(self):
        feats_list = []
        feats_dict = {}
        with open(self.arkscp, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(' ')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def get_cmvn_dict(self):
        cmvn_reader = kaldiio.load_scp_sequential(self.cmvnscp)

        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def concat_frame(self, features):
        time_steps, features_dim = features.shape
        concated_features = np.zeros(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32)
        # middle part is just the uttarnce
        concated_features[:, self.left_context_width * features_dim:
                             (self.left_context_width + 1) * features_dim] = features

        for i in range(self.left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
            (self.left_context_width - i - 1) * features_dim:
            (self.left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(self.right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
            (self.right_context_width + i + 1) * features_dim:
            (self.right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        return concated_features

    def subsampling(self, features):
        if self.frame_rate != 10:
            interval = int(self.frame_rate / 10)
            temp_mat = [features[i]
                        for i in range(0, features.shape[0], interval)]
            subsampled_features = np.row_stack(temp_mat)
            return subsampled_features
        else:
            return features


class AudioDataset(myDataset):
    def __init__(self, config, dataset_type):
        super(AudioDataset, self).__init__(config, dataset_type)

        self.config = config
        self.text = os.path.join(config.__getattr__(dataset_type), 'text')
        self.utt2num_frames_txt = os.path.join(config.__getattr__(dataset_type), 'utt2num_frames')

        self.short_first = config.short_first

        # if self.config.encoding:
        self.unit2idx = get_dict_from_scp(self.vocab, int)  # same function as get self.utt2num_frames_dict
        self.idx2unit = dict([(i, c) for (i, c) in enumerate(self.unit2idx)])
        self.targets_dict = self.get_targets_dict()
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

        feats_scp = self.feats_dict[utt_id]
        seq = self.targets_dict[utt_id]

        targets = np.array(seq)
        features = kaldiio.load_mat(feats_scp)

        if self.apply_cmvn:
            spk_id = self.utt2spk[utt_id]
            stats = self.cmvn_stats_dict[spk_id]
            features = cmvn(features, stats)

        features = self.concat_frame(features)
        features = self.subsampling(features)

        # if features 长度 > max_input_length,只保留前面部分
        if features.shape[0] >= self.config.max_input_length:
            features = features[:self.config.max_input_length, ]
        if targets.shape[0] >= self.max_target_length:
            targets = targets[:self.max_target_length]

        inputs_length = np.array(features.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths

    def get_targets_dict(self):
        targets_dict = {}
        with codecs.open(self.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                contents = parts[1:]
                # if len(contents) < 0 or len(contents) > self.max_target_length:
                #     continue
                # if self.config.encoding:
                labels = self.encode(contents)
                # else:
                #     labels = [int(i) for i in contents]
                targets_dict[utt_id] = labels
        return targets_dict

    def encode(self, seq):

        return [self.unit2idx[unit] if unit in self.unit2idx else self.unit2idx['<unk>'] for unit in seq]

    def decode(self, seq, rm_blk=False):

        if rm_blk:
            return " ".join([self.idx2unit[int(i)] for i in seq if i > 0])
        return " ".join([self.idx2unit[int(i)] for i in seq])

    def check_speech_and_text(self):
        featslist = copy.deepcopy(self.feats_list)
        for utt_id in featslist:
            if utt_id not in self.targets_dict:
                self.feats_list.remove(utt_id)


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

