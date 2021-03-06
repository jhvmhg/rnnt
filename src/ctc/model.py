import torch
import torch.nn as nn

from src.utils.build_model import build_encoder


class CTC(nn.Module):
    def __init__(self, config):
        super(CTC, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        self.project_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.config.enc.output_size, self.config.vocab_size)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.ctc_loss = nn.CTCLoss()

    def forward(self, inputs, inputs_length, targets, targets_length):
        enc_state, output_lengths = self.encoder(inputs, inputs_length)

        encoder_output = self.project_layer(enc_state)
        encoder_output = torch.transpose(encoder_output, 0, 1)
        encoder_output = encoder_output.log_softmax(2)

        loss = self.ctc_loss(encoder_output, targets.int(), output_lengths, targets_length.int())

        return loss

    def recognize(self, inputs, inputs_length):
        enc_states, output_lengths = self.encoder(inputs, inputs_length)
        encoder_output = self.project_layer(enc_states)

        preds = torch.argmax(encoder_output, -1)

        ans = [[int(j) for j in i if j > 0]
               for i in preds]

        return ans

    def get_post(self, inputs, inputs_length, apply_softmax=False):
        enc_states, output_lengths = self.encoder(inputs, inputs_length)
        encoder_output = self.project_layer(enc_states)
        if apply_softmax:
            encoder_output = self.softmax(encoder_output)

        return encoder_output, output_lengths
