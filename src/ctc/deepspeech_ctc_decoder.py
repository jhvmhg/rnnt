from itertools import groupby
import numpy as np
from math import log
import multiprocessing

def ctc_beam_search_decoder(probs_seq,
                            beam_size,
                            vocabulary,
                            cutoff_prob=1.0,
                            cutoff_top_n=40,
                            ext_scoring_func=None,
                            nproc=False):
    """CTC Beam search decoder.

    It utilizes beam search to approximately select top best decoding
    labels and returning results in the descending order.
    The implementation is based on Prefix Beam Search
    (https://arxiv.org/abs/1408.2873), and the unclear part is
    redesigned. Two important modifications: 1) in the iterative computation
    of probabilities, the assignment operation is changed to accumulation for
    one prefix may comes from different paths; 2) the if condition "if l^+ not
    in A_prev then" after probabilities' computation is deprecated for it is
    hard to understand and seems unnecessary.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param ext_scoring_func: External scoring function for
                            partially decoded sentence, e.g. word count
                            or language model.
    :type external_scoring_func: callable
    :param nproc: Whether the decoder used in multiprocesses.
    :type nproc: bool
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary):
            raise ValueError("The shape of prob_seq does not match with the "
                             "shape of the vocabulary.")

    # blank_id assign
    blank_id = 0

    # If the decoder called in the multiprocesses, then use the global scorer
    # instantiated in ctc_beam_search_decoder_batch().
    if nproc is True:
        global ext_nproc_scorer
        ext_scoring_func = ext_nproc_scorer

    ## initialize
    # prefix_set_prev: the set containing selected prefixes
    # probs_b_prev: prefixes' probability ending with blank in previous step
    # probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 1.0}
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    ## extend prefix in loop
    for time_step in range(len(probs_seq)):
        # prefix_set_next: the set containing candidate prefixes
        # probs_b_cur: prefixes' probability ending with blank in current step
        # probs_nb_cur: prefixes' probability ending with non-blank in current step
        prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        #If pruning is enabled
        if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in range(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            cutoff_len = min(cutoff_len, cutoff_top_n)
            prob_idx = prob_idx[0:cutoff_len]

        for l in prefix_set_prev:
            if not l in prefix_set_next:
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering prob_idx
            for index in range(cutoff_len):
                c, prob_c = prob_idx[index][0], prob_idx[index][1]

                if c == blank_id:
                    probs_b_cur[l] += prob_c * (
                        probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if l_plus not in prefix_set_next:
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                        probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 1.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix)
                        probs_nb_cur[l_plus] += score * prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        probs_nb_cur[l_plus] += prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = probs_nb_cur[
                        l_plus] + probs_b_cur[l_plus]
            # add l into prefix_set_next
            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
        # update probs
        probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

        ## store top beam_size prefixes
        prefix_set_prev = sorted(
            prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for seq, prob in prefix_set_prev.items():
        if prob > 0.0 and len(seq) > 1:
            result = seq[1:]
            # score last word by external scorer
            if (ext_scoring_func is not None) and (result[-1] != ' '):
                prob = prob * ext_scoring_func(result)
            log_prob = log(prob)
            beam_result.append((log_prob, result))
        else:
            beam_result.append((float('-inf'), ''))

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result

def ctc_beam_search_decoder_batch(probs_split,
                                  beam_size,
                                  vocabulary,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  cutoff_top_n=40,
                                  ext_scoring_func=None):
    """CTC beam search decoder using multiple processes.

    :param probs_seq: 3-D list with each element as an instance of 2-D list
                      of probabilities used by ctc_beam_search_decoder().
    :type probs_seq: 3-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param ext_scoring_func: External scoring function for
                            partially decoded sentence, e.g. word count
                            or language model.
    :type external_scoring_function: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    if not num_processes > 0:
        raise ValueError("Number of processes must be positive!")

    # use global variable to pass the externnal scorer to beam search decoder
    global ext_nproc_scorer
    ext_nproc_scorer = ext_scoring_func
    nproc = True

    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for i, probs_list in enumerate(probs_split):
        args = (probs_list, beam_size, vocabulary, cutoff_prob, cutoff_top_n,
                None, nproc)
        results.append(pool.apply_async(ctc_beam_search_decoder, args))

    pool.close()
    pool.join()
    beam_search_results = [result.get() for result in results]
    return beam_search_results

import kenlm
import os

class Scorer(object):
    """External scorer to evaluate a prefix or whole sentence in
       beam search decoding, including the score from n-gram language
       model and word count.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def _language_model_score(self, sentence):
        #log10 prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def _word_count(self, sentence):
        words = sentence.strip().split(' ')
        return len(words)

    # reset alpha and beta
    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    # execute evaluation
    def __call__(self, sentence, log=False):
        """Evaluation function, gathering all the different scores
        and return the final one.

        :param sentence: The input sentence for evalutation
        :type sentence: basestring
        :param log: Whether return the score in log representation.
        :type log: bool
        :return: Evaluation score, in the decimal or log.
        :rtype: float
        """
        lm = self._language_model_score(sentence)
        word_cnt = self._word_count(sentence)
        if log == False:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score

"""
使用：
dp_scoree = Scorer(0.5, 4.55, '/home1/meichaoyang/dataset/magictang/rnn_t/text_lm.apra')
ctc_score = ctc_beam_search_decoder(encoder_output[1], 10, list(dev_dataset.unit2idx),ext_scoring_func=dp_scoree)
"""