#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import torch
from six.moves import xrange

from src.data.utils import get_dict_from_scp


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, vocab, model, blank_index=0):
        self.model = model
        self.unit2idx = get_dict_from_scp(vocab, int)
        self.labels = list(self.unit2idx)
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(self.labels)])
        self.blank_index = blank_index
        space_index = len(self.labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in self.labels:
            space_index = self.labels.index(' ')
        self.space_index = space_index

    def decode(self, inputs, inputs_length):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            inputs: Tensor of fbank feature T * L * D
            inputs_length(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    """
    language model train by `awk '{$1="";print}' text > text_lm
                            lmplz -o 4 -S50% --interpolate_unigrams 0  <text_lm > text_lm.apra`
                            or `awk '{$1="";print}' text | lmplz -o 4 -S50% --interpolate_unigrams 0  > text_lm.apra
    """

    def __init__(self,
                 vocab,
                 model,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_index=0,
                 log_probs_input=False):
        super(BeamCTCDecoder, self).__init__(vocab, model, blank_index=0)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        # labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = CTCBeamDecoder(self.labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index, log_probs_input)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):

        """
        convert torch.tensor to list
        """
        # results = [[k.item() for k in j[0][:sizes[i][0]]] for i, j in enumerate(offsets)]
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size].cpu().numpy().tolist())
                else:
                    utterances.append([])
            results.append(utterances)
        return results

    def decode(self, inputs, inputs_length):

        """
        Decodes probability output using ctcdecode package.
        Arguments:
            inputs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            inputs_length: Size of each sequence in the mini-batch
        Returns:
            results_strings: sequences of the model's n-best guess for the transcription
                             Shape: batchsize x num_beams x num_timesteps.
            results_list: sequences of the model's n-best guess for the list(int)
                          Shape: batchsize x num_beams x num_timesteps.
        """

        encoder_output, output_lengths = self.model.get_post(inputs, inputs_length)
        probs, sizes = encoder_output.cpu(), output_lengths.cpu()
        results_tensor, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        results_strings = self.convert_to_strings(results_tensor, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        results_list = self.convert_tensor(results_tensor, seq_lens)

        return results_strings, results_list, scores, offsets


class GreedyDecoder(Decoder):
    def __init__(self, vocab, blank_index=0):
        super(GreedyDecoder, self).__init__(vocab, blank_index)

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets


def build_ctc_beam_decoder(config, model):
    beamctc_decoder = None
    if config.model.type == "ctc" and config.evaling.lm_model:
        alpha = config.evaling.alpha if config.evaling.alpha else 0.5
        beta = config.evaling.alpha if config.evaling.beta else 0.5
        cutoff_top_n = config.evaling.cutoff_top_n if config.evaling.cutoff_top_n else 40
        cutoff_prob = config.evaling.cutoff_prob if config.evaling.cutoff_prob else 1.0
        beam_width = config.evaling.beam_width if config.evaling.beam_width else 20
        log_probs_input = config.evaling.log_probs_input if config.evaling.log_probs_input else True
        num_processes = config.evaling.beam_decoder_num_processes if config.evaling.beam_decoder_num_processes else 4

        beamctc_decoder = BeamCTCDecoder(config.data.vocab, model, lm_path=config.evaling.lm_model,
                                         alpha=alpha,
                                         beta=beta,
                                         cutoff_top_n=cutoff_top_n,
                                         cutoff_prob=cutoff_prob,
                                         beam_width=beam_width,
                                         log_probs_input=log_probs_input,
                                         num_processes=num_processes
                                         )
    return beamctc_decoder
