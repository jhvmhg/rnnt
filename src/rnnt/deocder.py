import torch


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

    def forward(self, tn_output):

        hyps = self.transducer_greedy_decode(tn_output)
        return hyps

    def transducer_greedy_decode(self, tn_output):

        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        input_PN = (
                torch.ones(
                    (tn_output.size(0), 1),
                    device=tn_output.device,
                    dtype=torch.int64,
                )
                * self.blank_id
        )
        # First forward-pass on PN
        out_PN, hidden = self.model.decoder(input_PN)
        # For each time step
        for t_step in range(tn_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            log_probs = self.model.joint(
                tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
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
                    input_PN[i][0] = positions[i]
                    have_update_hyp.append(i)
            if len(have_update_hyp) > 0:
                # Select sentence to update
                # And do a forward steps + generated hidden
                (
                    selected_input_PN,
                    selected_hidden,
                ) = self._get_sentence_to_update(
                    have_update_hyp, input_PN, hidden
                )
                selected_out_PN, selected_hidden = self.model.decoder(
                    selected_input_PN, hidden=selected_hidden
                )
                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = self._update_hiddens(
                    have_update_hyp, selected_hidden, hidden
                )

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):

        selected_output_PN = output_PN[selected_sentences, :]
        # for LSTM hiddens (hn, hc)
        if isinstance(hidden, tuple):
            hidden0_update_hyp = hidden[0][:, selected_sentences, :]
            hidden1_update_hyp = hidden[1][:, selected_sentences, :]
            hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
        else:
            hidden_update_hyp = hidden[:, selected_sentences, :]
        return selected_output_PN, hidden_update_hyp

    def _update_hiddens(self, selected_sentences, updated_hidden, hidden):

        if isinstance(hidden, tuple):
            hidden[0][:, selected_sentences, :] = updated_hidden[0]
            hidden[1][:, selected_sentences, :] = updated_hidden[1]
        else:
            hidden[:, selected_sentences, :] = updated_hidden
        return hidden
