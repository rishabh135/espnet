import torch

from espnet2.diar.attractor.abs_attractor import AbsAttractor


class RnnAttractor(AbsAttractor):
    """encoder decoder attractor for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
        layer: int = 1,
        unit: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attractor_encoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )
        self.attractor_decoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )

        self.linear_projection = torch.nn.Linear(unit, 1)

    def forward(
        self, enc_input: torch.Tensor, ilens: torch.Tensor, dec_input: torch.Tensor
    ):
        """Forward.

        Args:
            enc_input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            dec_input (torch.Tensor): decoder input (zeros) [Batch, num_spk + 1, F]

        Returns:
            attractor: [Batch, num_spk + 1, F]
            att_prob: [Batch, num_spk + 1, 1]
        """
        _, hs = self.attractor_encoder(enc_input)
        attractor, _ = self.attractor_decoder(dec_input, hs)
        att_prob = self.linear_projection(attractor)
        return attractor, att_prob


class RnnAttractorNoGrad(AbsAttractor):
    """encoder decoder attractor for speaker diarization.
    
    This model will update only the fully connected (linear_projection) layer
    """

    def __init__(
        self,
        encoder_output_size: int,
        layer: int = 1,
        unit: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attractor_encoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )
        self.attractor_decoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )

        self.linear_projection = torch.nn.Linear(unit, 1)

    def forward(
        self, enc_input: torch.Tensor, ilens: torch.Tensor, dec_input: torch.Tensor
    ):
        """Forward.

        Args:
            enc_input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            dec_input (torch.Tensor): decoder input (zeros) [Batch, num_spk + 1, F]

        Returns:
            attractor: [Batch, num_spk + 1, F]
            att_prob: [Batch, num_spk + 1, 1]
        """
        _, hs = self.attractor_encoder(enc_input)
        attractor, _ = self.attractor_decoder(dec_input, hs)
        att_prob = self.linear_projection(attractor.detach())
        return attractor, att_prob
