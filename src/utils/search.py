import torch
import torch.nn.functional as F


def GreedyDecode(model, inputs, input_lengths):

    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)

    zero_token = torch.LongTensor([[0]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
    results = []
    batch_size = inputs.size(0)

    def decode(inputs, lengths):
        log_prob = 0
        token_list = []
        gu, hidden = model.decoder(zero_token)
        for t in range(lengths):
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            prob, pred = torch.max(out, dim=0)
            pred = int(pred.item())
            log_prob += prob.item()
            if pred != 0:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = model.decoder(token, hidden=hidden)

        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results


def BeamDecode(model, inputs, input_lengths):
    batch_size = inputs.size(0)

    enc_states, outputs_length = model.encoder(inputs, input_lengths)

    zero_token = torch.LongTensor([[0]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()

    def decode(enc_state, lengths, beam=10):
        hyps = [[]] * lengths

        dec_state, hidden = model.decoder(zero_token)

        for t in range(lengths):
            hyps_old = hyps[t]

            for j, hyp_old in enumerate(hyps_old):
                old_state = hyp_old['de_hidden']
                old_id = hyp_old['id']
                old_score = hyp_old['score']

                logits = model.joint(enc_state[t].view(-1), old_state.view(-1))
                out_probs = F.softmax(logits, dim=0).detach()

                # for k in range(beam):

                # hyps_best_kept = []
                # token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                dec_state, hidden = model.decoder(token, hidden=hidden)

                new_hyp = {}

                # new_hyp['de_hidden'] = h_list[:]
                # new_hyp['id'] = c_list[:]
                # new_hyp['score'] = hyp['score'] + local_best_scores[0, j]

                # will be (2 x beam) hyps at most
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept,
                                    key=lambda x: x['score'],
                                    reverse=True)[:beam]
            hyps[t] = hyps_best_kept

        # return token_list

    results = []
    for i in range(batch_size):
        decoded_seq = decode(enc_states[i], outputs_length[i])
        results.append(decoded_seq)

    return results



