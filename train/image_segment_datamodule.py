import csv
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

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
            clas = int(row[2])
            circum = float(row[3])
            opt_circum = float(row[4])
            
            # return values
            ret = {}
            ret['image'] = image_path
            ret['filepath'] = image_path
            ret['mask'] = mask_path
            ret['clas'] = clas
            ret['circum'] = circum
            ret['opt_circum'] = opt_circum
            rets.append(ret)
            
    return rets
    

class ImageSegmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(ImageSegmentDataModule, self).__init__()

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
        train_datalist = load_csv(self.cfg.Data.dataset.train_datalist,
                                  self.cfg.Data.dataset.top_dir)
        self.train_datalist = train_datalist
        valid_datalist = load_csv(self.cfg.Data.dataset.valid_datalist,
                                  self.cfg.Data.dataset.top_dir)
        if self.cfg.General.mode == "predict":
            predict_datalist = load_csv(self.cfg.Data.dataset.predict_datalist,
                                        self.cfg.Data.dataset.top_dir)
        # number of training data
        num_train_data = [0] * self.cfg.Data.dataset.num_classes
        for train_data in train_datalist:
            clas = train_data["clas"]
            num_train_data[clas] += 1
        self.num_train_data = num_train_data

        train_transforms = get_transform(self.cfg.Transform.train)
        valid_transforms = get_transform(self.cfg.Transform.valid)
        if self.cfg.General.mode == "predict":
            predict_transforms = get_transform(self.cfg.Transform.predict)
            
        self.dataset['train'] = Dataset(
            data=train_datalist,
            transform=train_transforms,
        )
        self.dataset['valid'] = Dataset(
            data=valid_datalist,
            transform=valid_transforms,
        )            

        if self.cfg.General.mode == "predict":
            self.dataset['predict'] = Dataset(
                data=predict_datalist,
                transform=predict_transforms,
            )
        
    def train_dataloader(self):
        num_total_data = 0
        for num_data in self.num_train_data:
            num_total_data += num_data
        weights = []
        for num_data in self.num_train_data:
            weights.append(num_total_data / num_data)
        print("num_total_data:", num_total_data)
        print("sampler weights:", weights)
            
        datalist = self.train_datalist
        sample_weights = [weights[datalist[idx]['clas']]
                          for idx in range(len(datalist))]        

        num_small_data = min(self.num_train_data)
        sampler = WeightedRandomSampler(sample_weights, 
                                        num_small_data * 2,
                                        replacement=False)
        
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.cfg.Data.dataloader.batch_size,
#            shuffle=self.cfg.Data.dataloader.train.shuffle,
            shuffle=False,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=False,
            collate_fn=list_data_collate,
            sampler=sampler,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset['valid'],
            batch_size=self.cfg.Data.dataloader.val_batch_size,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=False
        )
        return val_loader

    def predict_dataloader(self):
        if self.cfg.General.mode == "predict":
            predict_loader = DataLoader(
                self.dataset['predict'],
                batch_size=self.cfg.Data.dataloader.val_batch_size,
                shuffle=self.cfg.Data.dataloader.predict.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False
            )
        else:
            predict_loader = None
        return predict_loader

    # For validation of EMA model
    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset['valid'][:100],
            batch_size=self.cfg.Data.dataloader.val_batch_size,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=False
        )
        return test_loader
    
