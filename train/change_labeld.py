from monai.transforms import MapTransform
import numpy as np
import torch


class ChangeLabeld(MapTransform):
    def __init__(self, keys):
        assert(len(keys) == 1)
        super(ChangeLabeld, self).__init__(keys)

    def __call__(self, data):
        d = dict(data)

        # 
        mask = d[self.keys[0]]

        # change labels (binary)
        mask[mask == 2] = 1

        d[self.keys[0]] = mask
        
        return d
