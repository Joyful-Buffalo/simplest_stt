from typing import Union,Tuple

import torch


def _pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value

def _conv_output_length_int(
                length: int, 
                kernel_size: int, 
                stride: int=1, 
                padding: int=0, 
                dilation: int=1) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def _conv_output_length_tensor(
        length:torch.Tensor,
        kernel_size: int,
        stride: int=1,
        padding: int=0,
        dilation: int=1) -> torch.Tensor:
    length = length.to(dtype=torch.int64)
    out = torch.div(
        length + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode='floor') + 1
    return torch.clamp(out, min=0)