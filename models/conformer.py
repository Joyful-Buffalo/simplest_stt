from typing import Union
from torch import nn
import torchaudio

from models.utils import _conv_output_length_int, _pair, _conv_output_length_tensor



class Conv2dSubsampling(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int,
            kernel_size: Union[int,tuple[int, int]] = 3,
            stride: Union[int,tuple[int, int]] = 2,
            padding: Union[int,tuple[int, int]] = 0,
            dilation: Union[int,tuple[int, int]] = 1,
            dropout_rate: float = 0.0):
        super(Conv2dSubsampling, self).__init__()

        k_t, k_f = _pair(kernel_size)
        s_t, s_f = _pair(stride)
        p_t, p_f = _pair(padding)
        d_t, d_f = _pair(dilation)

        self.k_t, self.k_f = k_t, k_f
        self.s_t, self.s_f = s_t, s_f
        self.p_t, self.p_f = p_t, p_f
        self.d_t, self.d_f = d_t, d_f

        self.conv = nn.Sequential(
            nn.Conv2d(
                1, 
                out_dim, 
                kernel_size=(k_t, k_f), 
                stride=(s_t, s_f), 
                padding=(p_t, p_f), 
                dilation=(d_t, d_f)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                out_dim, 
                out_dim, 
                kernel_size=(k_t, k_f), 
                stride=(s_t, s_f), 
                padding=(p_t, p_f), 
                dilation=(d_t, d_f)),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        out_freq = _conv_output_length_int(in_dim, k_f, s_f, p_f, d_f)
        out_freq = _conv_output_length_int(out_freq, k_f, s_f, p_f, d_f)
        assert out_freq > 0, "Conv2dSubsampling reduces frequency dimension to zero or negative value" 
        self.out = nn.Linear(out_dim * out_freq, out_dim)

    def forward(self, x: nn.Tensor, x_len: nn.Tensor) -> tuple[nn.Tensor, nn.Tensor]:
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, time, feature)
        x = self.conv(x)    # (batch, out_dim, time', feature')

        _, out_dim, out_time, out_freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time', out_dim, feature')
        x = x.view(batch_size, out_time, out_dim * out_freq)  # (batch, time', out_dim * feature')

        x = self.out(x)  # (batch, time', out_dim)
        out_len = _conv_output_length_tensor(
            x_len,
            self.k_t,
            self.s_t,
            self.p_t,
            self.d_t)
        out_len = _conv_output_length_tensor(
            out_len,
            self.k_t,
            self.s_t,
            self.p_t,
            self.d_t)
        return x, out_len


class CTCConformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            vocab_size: int,
            encoder_dim: int,
            ffn_dim: int,
            num_layers: int,
            num_heads: int,
            dropout_rate: float,
            depthwise_conv_kernel_size: int = 31
            ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(
            in_dim=input_dim,
            out_dim=encoder_dim,
            dropout_rate=dropout_rate
        )
        self.conformer = torchaudio.models.Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size
        )
        self.ctc_linear = nn.Linear(encoder_dim, vocab_size)

    def forward(self, x: nn.Tensor, x_len: nn.Tensor) -> tuple[nn.Tensor, nn.Tensor]:
        x, x_len = self.subsampling(x, x_len)
        x = self.conformer(x, x_len)
        x = self.ctc_linear(x)
        return x, x_len