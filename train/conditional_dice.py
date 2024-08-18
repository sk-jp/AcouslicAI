import torch
from torch import Tensor

from torchmetrics import Dice


class ConditionalDice(Dice):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        index = []
        for b in range(preds.shape[0]):
            if torch.any(preds[b] > 0) or torch.any(target[b] > 0):
                index.append(b)
        
        if len(index) > 0:
            super().update(preds[index], target[index])
        

if __name__ == "__main__":
    from torch import tensor
    
    preds1 = tensor([[0, 0, 0, 2],
                     [0, 0, 0, 0]])
    target1 = tensor([[0, 0, 0, 2],
                      [0, 0, 0, 0]])
    preds2 = tensor([[0, 0, 0, 2],
                     [0, 1, 0, 0]])
    target2 = tensor([[0, 0, 0, 0],
                      [0, 1, 1, 0]])

    # dice
    dice = ConditionalDice(num_classes=3,
                           average='micro',
                           ignore_index=0)
#    dice.update(preds1, target1)
    dice.update(preds2, target2)
    
    print(dice.compute())
    
    