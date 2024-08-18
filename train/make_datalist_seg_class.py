import cv2
import numpy as np
import pandas as pd

from ellipse_circumference import ellipse_circumference

phase = "train"
#phase = "valid"
train_ratio = 0.8

num_slices = 3

topdir = "/data/MICCAI2024_ACOUSLIC_AI"
decoded_dir = f"{topdir}/decoded_data"
image_dir = f"{decoded_dir}/images"
mask_dir = f"{decoded_dir}/masks"
circum_file = f"{topdir}/circumferences/fetal_abdominal_circumferences_per_sweep.csv"

# read circum_files
circums = pd.read_csv(circum_file, header=0)

num_images_all = 840
num_scans = 6
num_images_in_scan = num_images_all // num_scans
mm_pixel = 0.28 # mm/pixel

num_train_cases = int(len(circums) * train_ratio)

for index, row in circums.iterrows():
    if phase == "train" and index >= num_train_cases:
        continue
    if phase == "valid" and index < num_train_cases:
        continue
        
    uuid = row['uuid']
    
    for scan in range(num_scans):
        for z in range(num_images_in_scan):
            # image path and mask path
            image_files = []
            for n in range(-(num_slices//2), num_slices//2 + 1):
                sl = z + n
                if sl < 0:
                    sl = 0
                elif sl >= num_images_in_scan:
                    sl = num_images_in_scan - 1
                    
                sl += scan * num_images_in_scan
                
                image_file = f"{image_dir}/image_{uuid}_{sl:03d}.png"
                image_files.append(image_file)
            
            sl = scan * num_images_in_scan + z
            mask_file = f"{mask_dir}/mask_{uuid}_{sl:03d}.png"
            # read mask file
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # check mask
            if mask.max() == 0:
                # circum
                circum = 0.0
                # class label
                clas = 0
            else:
                # measure the circum
                # get contours
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                # select maximum contour
                max_contour = max(contours, key=lambda x: cv2.contourArea(x))
                # fit ellipse
                (cx, cy), (width, height), angle = cv2.fitEllipse(max_contour)           
                # get arc length
                a = max(width/2, height/2)
                b = min(width/2, height/2)
                circum = ellipse_circumference(a, b)
                circum = circum * mm_pixel
                # class label
                clas = 1

            # opt/subopt circum
            opt_circum = row[f"sweep_{scan+1}_ac_mm"]
            
            # print
            s = ""
            for image_file in image_files:
                image_file = image_file.replace(f"{topdir}/", "")
                s += f"{image_file}," 
            mask_file = mask_file.replace(f"{topdir}/", "")
            s += f"{mask_file}"
            
            if clas == 0:
                print(f"{s},{clas},{circum},0.0")
            else:
                print(f"{s},{clas},{circum},{opt_circum}")
