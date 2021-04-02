import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.build_model import build_encoder, build_decoder
from warprnnt_pytorch import RNNTLoss
from src.net.loss import nll_loss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size, joint="concat"):
        super(JointNet, self).__init__()
        self.joint = joint
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_size, inner_dim, bias=True),
            nn.Tanh(),
            nn.Linear(inner_dim, vocab_size, bias=True)
        )

    def forward(self, enc_state, dec_state, softmax=False):
        """Returns the fusion of inputs tensors.

        Arguments
        ---------
        enc_state : torch.Tensor (B * T * H_t) or ([B *] H_t)
           Input from Transcription Network.

        dec_state : torch.Tensor (B * U * H_p) or ([B *] H_p)
           Input from Prediction Network.

        softmax : bool
           apply softmax for joint output.
        """

        if enc_state.dim() != dec_state.dim():
            raise ValueError("input_TN and input_PN must be have same size."
                             "3 for training, or 1,2 for evaluation")

        if enc_state.dim() == 3 and dec_state.dim() == 3:
            enc_state = enc_state.unsqueeze(2)
            dec_state = dec_state.unsqueeze(1)

        if self.joint == "sum":
            concat_state = enc_state + dec_state
        elif self.joint == "concat":
            if enc_state.dim() == 4 and dec_state.dim() == 4:
                t = enc_state.size(1)
                u = dec_state.size(2)
                enc_state = enc_state.repeat([1, 1, u, 1])
                dec_state = dec_state.repeat([1, t, 1, 1])
            else:
                assert enc_state.dim() == dec_state.dim()

            concat_state = torch.cat((enc_state, dec_state), dim=-1)
        else:
            raise NotImplementedError

        del enc_state, dec_state
        joint = self.mlp(concat_state)
        if softmax:
            joint = self.softmax(joint)

        return joint


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        self.config = config
        # define encoder
        self.encoder = build_encoder(config)
        # define decoder
        self.decoder = build_decoder(config)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size,
            joint=config.joint.type if config.joint.type else "concat"
        )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (
                self.decoder.embedding.weight.size(1), self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.transducer_loss = RNNTLoss()

        # multask learning (loss_decoder and loss_encoder)
        if config.enc.ctc_weight and config.enc.ctc_weight > 0.0:
            self.ctc_loss = nn.CTCLoss()
            self.encoder_project_layer = nn.Sequential(nn.Tanh(),
                                                       nn.Linear(self.config.enc.output_size, self.config.vocab_size))
        if config.dec.ce_weight and config.dec.ce_weight > 0.0:
            self.nll_loss = nll_loss
            self.decoder_project_layer = nn.Sequential(nn.Tanh(),
                                                       nn.Linear(self.config.dec.output_size, self.config.vocab_size))

    def forward(self, inputs, inputs_length, tokens, tokens_length, ctc_weight=0.0, ce_weight=0.0):
        """
            ctc_weight and ce_weight for multask learning
        """

        enc_state, enc_output_lengths = self.encoder(inputs, inputs_length)

        tokens_with_bos, token_with_bos_lens = F.pad(tokens, pad=[1, 0, 0, 0], value=0), tokens_length.add(1)
        tokens_with_eos, token_with_eos_lens = F.pad(tokens, pad=[0, 1, 0, 0], value=0), tokens_length.add(1)
        if enc_state.is_cuda:
            output_length, tokens_with_bos = enc_output_lengths.int().cuda(), tokens_with_bos.cuda()
        else:
            output_length = enc_output_lengths.int()

        # loss_transducer
        dec_state, _ = self.decoder(tokens_with_bos, tokens_length.add(1))
        p_transducer = self.joint(enc_state, dec_state)
        loss_transducer = self.transducer_loss(p_transducer, tokens.int(), output_length, tokens_length.int())

        # loss_decoder
        if self.ctc_loss and ctc_weight > 0.0:
            encoder_output = self.encoder_project_layer(enc_state)
            encoder_output = torch.transpose(encoder_output, 0, 1)
            encoder_output = encoder_output.log_softmax(2)

            loss_transducer += self.ctc_loss(encoder_output, tokens.int(),
                                             enc_output_lengths, tokens_length.int()) * ctc_weight
        # loss_encoder
        if self.nll_loss and ce_weight > 0.0:
            dec_output = self.decoder_project_layer(dec_state)
            dec_output = torch.nn.functional.log_softmax(dec_output, dim=-1)
            loss_transducer += self.nll_loss(dec_output, tokens_with_eos,
                                             length=token_with_eos_lens) * ce_weight

        return loss_transducer

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
        dec_states = self.decoder(inputs, inputs_length)[0]  # enc_states: N*T*D
        logits = self.project_layer(dec_states)  # logits: N*T*C
        logits = self.reshape_logits(logits, inputs_length)  # (N*T) * C
        targets = self.reshape_targets(targets, targets_length)  # (N*T)
        loss = self.crit(logits, targets)

        return loss

    def recognize(self, inputs, inputs_length):
        dec_states = self.decoder(inputs, inputs_length)[0]  # enc_states: N*T*D
        logits = self.project_layer(dec_states)  # logits: N*T*C

        preds = torch.argmax(logits, -1)

        ans = [j[:inputs_length[i].item()].cpu().numpy().tolist()
               for i, j in enumerate(preds)]

        return ans

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
        inputs_seq = torch.zeros((inputs_length.sum().item(), self.vocab_size),
                                 dtype=torch.float32)  # targets_length.sum()
        if logits.is_cuda:
            inputs_seq = inputs_seq.cuda()
        for i, b in enumerate(logits):
            inputs_seq.narrow(0, index, inputs_length[i]).copy_(b[:inputs_length[i], :])
            index += inputs_length[i]
        return inputs_seq
