import torch
import torch.nn as nn

from monai.losses import DiceLoss, HausdorffDTLoss, FocalLoss
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

from conditional_dice_loss import ConditionalDiceLoss


class Loss(_Loss):
    def __init__(self, conf):
        super(Loss, self).__init__()

        lossfun_names = conf.lossfuns
        weights = torch.tensor(conf.lossfun_weights)
        
        lossfuns = dict()
        lossfun_weights = dict()
        for idx, lossfun_name in enumerate(lossfun_names):
            lossfun = eval(lossfun_name)(**conf[lossfun_name].params)
            name = conf[lossfun_name].name
            lossfuns[name] = lossfun
            lossfun_weights[name] = weights[idx]
            
        self.lossfuns = lossfuns
        self.lossfun_weights = lossfun_weights

    def forward(self, pred, target):
        total_loss = 0
        losses = {}        
        
        for key in self.lossfuns.keys():
            loss = self.lossfuns[key](pred[key], target[key])
            
            device = pred[key].device
            total_loss += self.lossfun_weights[key].to(device) * loss
            losses[key] = loss
            
        return total_loss, losses
    
    