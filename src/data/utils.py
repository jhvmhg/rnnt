import numpy as np
import codecs
import kaldiio


def pad(inputs, max_length):
    """
    if inputs.shape[0] >= max_length , just return inputs[:max_length,]
    """
    dim = len(inputs.shape)
    if dim == 1:

        if inputs.shape[0] >= max_length:
            padded_inputs = inputs[:max_length]
        else:
            pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
    elif dim == 2:

        feature_dim = inputs.shape[1]
        if inputs.shape[0] >= max_length:
            padded_inputs = inputs[:max_length, ]
        else:
            pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
            padded_inputs = np.row_stack([inputs, pad_zeros_mat])
    else:
        raise AssertionError(
            'Features in inputs list must be one vector or two dimension matrix! ')
    return padded_inputs


def pad_np(inputs, max_length):
    """
    if inputs.shape[0] >= max_length , just return inputs[:max_length,]
    """
    dim = len(inputs[0].shape) + 1
    batch_zise = len(inputs)
    if dim == 2:  # batch_zise * target_len
        pad_zeros_mat = np.zeros([batch_zise, max_length], dtype=np.int64)
        for x in range(batch_zise):
            pad_zeros_mat[x, :len(inputs[x])] = inputs[x]

    elif dim == 3:  # batch_zise * feat_len * feature_dim
        feature_dim = len(inputs[0][0])
        pad_zeros_mat = np.zeros([batch_zise, max_length, feature_dim], dtype=np.float32)
        for x in range(batch_zise):
            pad_zeros_mat[x, :len(inputs[x]), :] = inputs[x]
    else:
        raise AssertionError(
            'Features in inputs list must be two vector or three dimension matrix! ')
    return pad_zeros_mat


def cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))


def get_dict_from_scp(vocab, func=int):
    unit2idx = {}
    with codecs.open(vocab, 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split()
            unit = parts[0]
            idx = func(parts[1])
            unit2idx[unit] = idx
    return unit2idx


def get_feats_list(arkscp):
    feats_list = []
    feats_dict = {}
    with open(arkscp, 'r') as fid:
        for line in fid:
            key, path = line.strip().split(' ')
            feats_list.append(key)
            feats_dict[key] = path
    return feats_list, feats_dict


def get_cmvn_dict(cmvnscp):
    cmvn_stats_dict = {}
    cmvn_reader = kaldiio.load_scp_sequential(cmvnscp)

    for spkid, stats in cmvn_reader:
        cmvn_stats_dict[spkid] = stats

    return cmvn_stats_dict


def concat_frame(features, left_context_width, right_context_width):
    time_steps, features_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, features_dim *
               (1 + left_context_width + right_context_width)],
        dtype=np.float32)
    # middle part is just the uttarnce
    concated_features[:, left_context_width * features_dim:
                         (left_context_width + 1) * features_dim] = features

    for i in range(left_context_width):
        # add left context
        concated_features[i + 1:time_steps,
        (left_context_width - i - 1) * features_dim:
        (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

    for i in range(right_context_width):
        # add right context
        concated_features[0:time_steps - i - 1,
        (right_context_width + i + 1) * features_dim:
        (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

    return concated_features


def subsampling(features, frame_rate):
    if frame_rate != 10:
        interval = int(frame_rate / 10)
        temp_mat = [features[i]
                    for i in range(0, features.shape[0], interval)]
        subsampled_features = np.row_stack(temp_mat)
        return subsampled_features
    else:
        return features
