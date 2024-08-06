import monai.transforms as mt
import numpy as np

# from change_labeld import ChangeLabeld


def get_transform(conf_augmentation):
    """ Get augmentation function
        Args:
            conf_augmentation (Dict): dictionary of augmentation parameters
    """
    def get_object(trans):
        if trans.name in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(mt, trans.name)(augs_tmp, **trans.params)

        if trans.name == "NormalizeIntensityd":
            trans.params.subtrahend = np.array(trans.params.subtrahend)
            trans.params.divisor = np.array(trans.params.divisor)

        if hasattr(mt, trans.name):
            return getattr(mt, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]

    return mt.Compose(augs)

