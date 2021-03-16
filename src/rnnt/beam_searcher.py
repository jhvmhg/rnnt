import torch
from src.rnnt.model import Transducer


def _update_hiddens(selected_sentences, updated_hidden, hidden):
    if isinstance(hidden, tuple):
        hidden[0][:, selected_sentences, :] = updated_hidden[0]
        hidden[1][:, selected_sentences, :] = updated_hidden[1]
    else:
        hidden[:, selected_sentences, :] = updated_hidden
    return hidden


def _get_sentence_to_update(selected_sentences, output_PN, hidden):
    selected_output_PN = output_PN[selected_sentences, :]
    # for LSTM hiddens (hn, hc)
    if isinstance(hidden, tuple):
        hidden0_update_hyp = hidden[0][:, selected_sentences, :]
        hidden1_update_hyp = hidden[1][:, selected_sentences, :]
        hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
    else:
        hidden_update_hyp = hidden[:, selected_sentences, :]
    return selected_output_PN, hidden_update_hyp


class TransducerBeamSearcher(torch.nn.Module):

    def __init__(
            self,
            model,
            beam_size=4,
            nbest=5,
            lm_module=None,
            blank_id=0,
            lm_weight=0.0,
            state_beam=2.3,
            expand_beam=2.3,
    ):
        super(TransducerBeamSearcher, self).__init__()
        if not isinstance(model, Transducer):
            raise NotImplementedError
        self.model = model
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        if self.beam_size <= 1:
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, inputs, inputs_length):
        tn_output = self.model.encoder(inputs, inputs_length)[0]
        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(self, encoder_output):

        hyp = {
            "prediction": [[] for _ in range(encoder_output.size(0))],
            "logp_scores": [0.0 for _ in range(encoder_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        input_decoder = (
                torch.ones(
                    (encoder_output.size(0), 1),
                    device=encoder_output.device,
                    dtype=torch.int64,
                )
                * self.blank_id
        )
        # First forward-pass on PN
        out_PN, hidden = self.model.decoder(input_decoder)
        # For each time step
        for t_step in range(encoder_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            log_probs = self.model.joint(
                encoder_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                out_PN.unsqueeze(1), True
            )
            # Sort outputs at time
            logp_targets, positions = torch.max(
                self.softmax(log_probs).squeeze(1).squeeze(1), dim=1
            )
            # Batch hidden update
            have_update_hyp = []
            for i in range(positions.size(0)):
                # Update hiddens only if
                # 1- current prediction is non blank
                if positions[i].item() != self.blank_id:
                    hyp["prediction"][i].append(positions[i].item())
                    hyp["logp_scores"][i] += logp_targets[i]
                    input_decoder[i][0] = positions[i]
                    have_update_hyp.append(i)
            if len(have_update_hyp) > 0:
                # Select sentence to update
                # And do a forward steps + generated hidden
                (
                    selected_input_PN,
                    selected_hidden,
                ) = _get_sentence_to_update(
                    have_update_hyp, input_decoder, hidden
                )
                selected_out_PN, selected_hidden = self.model.decoder(
                    selected_input_PN, hidden=selected_hidden
                )
                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = _update_hiddens(
                    have_update_hyp, selected_hidden, hidden
                )

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    def transducer_beam_search_decode(self, encoder_output):

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        for i_batch in range(encoder_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                    torch.ones((1, 1), device=encoder_output.device, dtype=torch.int64)
                    * self.blank_id
            )
            input_decoder = (
                    torch.ones((1, 1), device=encoder_output.device, dtype=torch.int64)
                    * self.blank_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.blank_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(encoder_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                                          / len(x["prediction"]),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_decoder[0, 0] = a_best_hyp["prediction"][-1]
                    out_PN, hidden = self.model.decoder(
                        input_decoder,
                        hidden=a_best_hyp["hidden_dec"],
                    )
                    # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                    log_probs = self.model.joint(
                        encoder_output[i_batch, t_step, :].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                        out_PN.unsqueeze(0), True
                    )

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_decoder, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    best_logp = (
                        logp_targets[0]
                        if positions[0] != blank
                        else logp_targets[1]
                    )

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                                          + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            topk_hyp["hidden_dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                        self.lm_weight
                                        * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)
        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            ).exp().mean(),
            nbest_batch,
            nbest_batch_score,
        )
