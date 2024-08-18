import csv
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from monai.data import (
    Dataset,
    list_data_collate,
)

from get_transform import get_transform


# load csv file
def load_csv(csv_file, topdir):    
    rets = []
    with open(csv_file) as f:
        rows = csv.reader(f)
        for row in rows:
            image_path = f"{topdir}/{row[0]}"
            mask_path = f"{topdir}/{row[1]}"
            circum = []
            for r in row[2:]:
                circum.append(float(r))
                
            # return values
            ret = {}
            ret['image'] = image_path
            ret['mask'] = mask_path
            ret['circum'] = circum
            ret['filepath'] = image_path
            rets.append(ret)
            
    return rets
    

class ImageSegmentMhaDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(ImageSegmentMhaDataModule, self).__init__()

        # configs
        self.cfg = cfg

    # called once from main process
    # called only within a single process
    def prepare_data(self):
        # prepare data
        pass
    
    # perform on every GPU
    def setup(self, stage):
        self.dataset = {}        
        if self.cfg.General.mode == "train":
            train_datalist = load_csv(self.cfg.Data.dataset.train_datalist,
                                      self.cfg.Data.dataset.top_dir)
            valid_datalist = load_csv(self.cfg.Data.dataset.valid_datalist,
                                      self.cfg.Data.dataset.top_dir)
            train_transforms = get_transform(self.cfg.Transform.train)
            valid_transforms = get_transform(self.cfg.Transform.valid)

            self.dataset['train'] = Dataset(
                data=train_datalist,
                transform=train_transforms,
            )
            self.dataset['valid'] = Dataset(
                data=valid_datalist,
                transform=valid_transforms,
            )            
        elif self.cfg.General.mode == "predict":
            predict_datalist = load_csv(self.cfg.Data.dataset.predict_datalist,
                                        self.cfg.Data.dataset.top_dir)
            predict_transforms = get_transform(self.cfg.Transform.predict)
            
            self.dataset['predict'] = Dataset(
                data=predict_datalist,
                transform=predict_transforms,
            )
        
    def train_dataloader(self):
        if self.cfg.General.mode == "train":
            train_loader = DataLoader(
                self.dataset['train'],
                batch_size=self.cfg.Data.dataloader.batch_size,
                shuffle=self.cfg.Data.dataloader.train.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False,
                collate_fn=list_data_collate,
            )
        else:
            train_loader = None
        return train_loader

    def val_dataloader(self):
        if self.cfg.General.mode == "train" or self.cfg.General.mode == "validate":
            val_loader = DataLoader(
                self.dataset['valid'],
                batch_size=self.cfg.Data.dataloader.val_batch_size,
                shuffle=self.cfg.Data.dataloader.valid.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False
            )
        else:
            val_loader = None
        return val_loader

    def predict_dataloader(self):
        if self.cfg.General.mode == "predict":
            predict_loader = DataLoader(
                self.dataset['predict'],
                batch_size=self.cfg.Data.dataloader.pred_batch_size,
                shuffle=self.cfg.Data.dataloader.predict.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False
            )
        else:
            predict_loader = None
        return predict_loader
