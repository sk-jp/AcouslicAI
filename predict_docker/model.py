from pathlib import Path

import numpy as np
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)

from monai.transforms import AsDiscrete
from calc_circum import calc_circum

RESOURCE_PATH = Path("resources")

class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self, model, transform, cfg):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        self.model = model
        self.transform = transform
        self.cfg = cfg
        self.post_pred = AsDiscrete(argmax=True, dim=1)
        
    def predict(self, input_img_path, save_probabilities=True):
        """
        Use trained network to generate segmentation masks
        """
        
        # read an image and preprocess 
        inputs = {}
        inputs["image"] = input_img_path
        image = self.transform(inputs)["image"]

        # check if GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        self.model.to(device)
        self.model.eval()
        
        # prediction
        print("Predicting ...")
        preds_mask = []
        preds_clas = []
        image = image.transpose(0, 1)   # [z,c,y,x]
        
        if self.cfg.Data.dataset.num_slices == 1:
            for zs in range(0, image.shape[0], self.cfg.Data.test.sub_batch_size):
                ze = zs + self.cfg.Data.test.sub_batch_size
                if ze > image.shape[0]:
                    ze = image.shape[0]
                with torch.no_grad():
                    slices = image[zs:ze].to(device)
                    pred_mask, pred_clas = self.model(slices)  # [b,c,y,x]
                preds_mask.append(pred_mask.cpu())
                preds_clas.append(pred_clas.cpu())
            pred_mask = torch.cat(preds_mask)
            pred_clas = torch.cat(preds_clas)
        elif self.cfg.Data.dataset.num_slices > 1:
            num_slices = self.cfg.Data.dataset.num_slices
            sub_batch_size = self.cfg.Data.test.sub_batch_size
            num_scans = self.cfg.Data.dataset.num_scans
            num_images_per_scan = self.cfg.Data.dataset.num_images_per_scan
                
            preds_mask = []
            preds_clas = []

            for scan in range(num_scans):
                image_scan = image[num_images_per_scan * scan: num_images_per_scan * (scan + 1), ...]
                image_add_0 = image_scan[0, ...].unsqueeze(0)
                image_add_1 = image_scan[-1, ...].unsqueeze(0)
                image_cat = []
                for z in range(num_slices // 2):
                    image_cat.append(image_add_0)
                image_cat.append(image_scan)
                for z in range(num_slices // 2):
                    image_cat.append(image_add_1)
                image_scan = torch.cat(image_cat)
                
                for zs in range(0, num_images_per_scan, sub_batch_size):
                    ze = zs + sub_batch_size
                    if ze > num_images_per_scan:
                        ze = num_images_per_scan
                    image_batch = []
                    for z in range(zs, ze):
                        ss = z
                        se = z + num_slices
                        if se > num_images_per_scan + num_slices - 1:
                            se = num_images_per_scan + num_slices - 1
                            
                        image_batch.append(image_scan[ss:se, 0, ...])
                    image_batch = torch.stack(image_batch)
                    
                    with torch.no_grad():
                        image_batch = image_batch.to(device)
                        pred_mask, pred_clas = self.model(image_batch)  # [b,n,y,x]
                    
                    preds_mask.append(pred_mask.cpu())
                    preds_clas.append(pred_clas.cpu())
                    
            pred_mask = torch.cat(preds_mask)
            pred_clas = torch.cat(preds_clas)

        # argmax
        pred_mask = self.post_pred(pred_mask)    # [z,c=1,y,x]
        pred_clas = self.post_pred(pred_clas)    # [z,c=1]
            
        # check each slice
        print("Postprocessing ...")
        mm_pixel = 0.28 / self.cfg.Transform.resize_ratio
        argmax_z = -1
#        max_circ = -1
        max_pred_cls_one = -1

        num_images_per_scan = self.cfg.Data.dataset.num_images_per_scan
        for scan in range(self.cfg.Data.dataset.num_scans):
            argmax_pos_in_scan = -1
            max_circ_in_scan = -1
            pred_cls_one = 0
            for pos in range(num_images_per_scan):
                z = scan * num_images_per_scan + pos
                if pred_clas[z] == 1:
                    pred_circ, _ = calc_circum(pred_mask[z][0], mm_pixel)
                    
                    if pred_circ > max_circ_in_scan:
                        max_circ_in_scan = pred_circ
                        argmax_pos_in_scan = z
                        
                    pred_cls_one += 1
                        
            if pred_cls_one > max_pred_cls_one:
                max_pred_cls_one = pred_cls_one
#                max_circ = max_circ_in_scan
                argmax_z = argmax_pos_in_scan
                
            print(f"  scan{scan}: pred_cls_one={pred_cls_one}, circ={max_circ_in_scan:.3f}")

        if argmax_z == -1:
            height, width = pred_mask.shape[2:]
            selected_mask = np.zeros((height, width), dtype=np.uint8)
            selected_frame_number = argmax_z
        else:
            selected_mask = pred_mask[argmax_z][0].numpy()
            selected_mask = selected_mask.astype(np.uint8)
            selected_frame_number = argmax_z

        return selected_mask, selected_frame_number
        
