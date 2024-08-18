from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchseg


class UnetMultitask(nn.Module):
    def __init__(
        self,
        encoder_name: str = "convnextv2_tiny",
        in_channels: int = 1,
        classes: int = 2,
        encoder_weights: bool = True,
        encoder_depth: int = 4,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        decoder_use_batchnorm: bool = True,             # True, False, "inplace"
        decoder_attention_type: Optional[str] = None,   # None, "scse"
        aux_pooling: str = "avg",       # "avg", "max"
        aux_dropout: float = None,      # None, float
        aux_activation: Callable = nn.Identity(),   # nn.Identity(), nn.Sigmoid(), nn.Softmax()
    ) -> None:
        super(UnetMultitask, self).__init__()

        if "convnext" in encoder_name:
            head_upsampling = 2
        else:
            head_upsampling = 1
            
        aux_params = dict(
            pooling = aux_pooling,         # one of 'avg', 'max'
            dropout = aux_dropout,         # dropout ratio, default is None
            activation = aux_activation,   # activation function, default is Identity
            classes=classes,               # define number of output labels
        )

        self.model = torchseg.Unet(
            encoder_name,
            in_channels=in_channels,
            classes=classes,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            head_upsampling=head_upsampling,
            aux_params=aux_params,
        )

    def forward(self, x) -> tuple:
        y_mask, y_clas = self.model(x)

        return y_mask, y_clas

if __name__ == "__main__":
    model = UnetMultitask("convnextv2_tiny", 1, 2)
    x = torch.rand((4, 1, 256, 256), dtype=torch.float32)
    y_mask, y_clas = model(x)
    print("y_mask:", y_mask.shape)
    print("y_clas:", y_clas.shape)
