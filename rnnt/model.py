import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.build_model import build_encoder, build_decoder
from warprnnt_pytorch import RNNTLoss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, inner_dim, bias=True),
            nn.Tanh(),
            nn.Linear(inner_dim, vocab_size, bias=True)
        )

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.mlp(concat_state)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        # define decoder
        self.decoder = build_decoder(config)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
        )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (
            self.decoder.embedding.weight.size(1), self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.crit = RNNTLoss()

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, output_length = self.encoder(inputs, inputs_length)
        if enc_state.is_cuda:
            output_length = output_length.int().cuda()
        else:
            output_length = output_length.int()
        concat_targets = F.pad(targets, pad=[1, 0, 0, 0], value=0)

        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)

        loss = self.crit(logits, targets.int(), output_length, targets_length.int())

        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states, outputs_length = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], outputs_length[i])
            results.append(decoded_seq)

        return results


class LM(nn.Module):
    def __init__(self, config):
        super(LM, self).__init__()
        self.config = config
        self.vocab_size = self.config.vocab_size
        # define decoder
        self.decoder = build_decoder(config)
        # define project_layer
        self.project_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.config.dec.output_size, self.config.vocab_size)
        )

        self.crit = nn.CrossEntropyLoss()

    def forward(self, inputs, inputs_length, targets, targets_length):
        """
        inputs: N*T
        targets: N*T
        """
        enc_states = self.decoder(inputs, inputs_length)[0]  # enc_states: N*T*D
        logits = self.project_layer(enc_states)  # logits: N*T*C
        logits = self.reshape_logits(logits, inputs_length) # (N*T) * C
        targets = self.reshape_targets(targets, targets_length) # (N*T)
        loss = self.crit(logits, targets)

        return loss

    def reshape_targets(self, targets, targets_length):
        index = 0
        targets_seq = torch.zeros(targets_length.sum(), dtype=torch.int64)  # targets_length.sum()
        if targets.is_cuda:
            targets_seq = targets_seq.cuda()
        for i, b in enumerate(targets):
            targets_seq.narrow(0, index, targets_length[i]).copy_(b[:targets_length[i]])
            index += targets_length[i]
        return targets_seq

    def reshape_logits(self, logits, inputs_length):
        index = 0
        inputs_seq = torch.zeros((inputs_length.sum().item(), self.vocab_size), dtype=torch.float32)  # targets_length.sum()
        if logits.is_cuda:
            inputs_seq = inputs_seq.cuda()
        for i, b in enumerate(logits):
            inputs_seq.narrow(0, index, inputs_length[i]).copy_(b[:inputs_length[i], :])
            index += inputs_length[i]
        return inputs_seq
