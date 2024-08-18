import torch
from torch import Tensor
import warnings

from monai.losses import DiceLoss


class ConditionalDiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_am = torch.argmax(input, 1)
        
        index = []
        for b in range(input_am.shape[0]):
            if torch.any(input_am[b] > 0) or torch.any(target[b] > 0):
                index.append(b)
        
        if len(index) > 0:
            f = super().forward(input[index], target[index])
        else:
            f = 0
            
        return f
        
