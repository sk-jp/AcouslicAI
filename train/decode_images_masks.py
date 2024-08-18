import cv2
import pandas as pd
import SimpleITK as sitk


topdir = "/data/MICCAI2024_ACOUSLIC_AI"
image_dir = f"{topdir}/images/stacked_fetal_ultrasound"
mask_dir = f"{topdir}/masks/stacked_fetal_abdomen"
circum_file = f"{topdir}/circumferences/fetal_abdominal_circumferences_per_sweep.csv"
output_dir = f"{topdir}/decoded_data"
output_list_file = f"{output_dir}/datalist.csv"
output_list = []

# read circum_files
circums = pd.read_csv(circum_file, header=0)

for index, row in circums.iterrows():
    uuid = row['uuid']
    print("uuid:", uuid)
    
    # image
    image_file = f"{image_dir}/{uuid}.mha"
    image_reader = sitk.ImageFileReader()
    image_reader.SetImageIO("MetaImageIO")
    image_reader.SetFileName(image_file)
    image_reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    image_reader.ReadImageInformation()  # Get just the information from the file

    xdim, ydim, zdim = image_reader.GetSize()
    xres, yres, zres = image_reader.GetSpacing() 

    images = image_reader.Execute()
    images = sitk.GetArrayFromImage(images)   # numpy

    # mask
    mask_file = f"{mask_dir}/{uuid}.mha"
    mask_reader = sitk.ImageFileReader()
    mask_reader.SetImageIO("MetaImageIO")
    mask_reader.SetFileName(mask_file)
    mask_reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    mask_reader.ReadImageInformation()  # Get just the information from the file

    xdim, ydim, zdim = mask_reader.GetSize()
    xres, yres, zres = mask_reader.GetSpacing() 

    masks = mask_reader.Execute()
    masks = sitk.GetArrayFromImage(masks)   # numpy

    # check the masks
    num_images = images.shape[0]
    for z in range(num_images):
        image = images[z]
        mask = masks[z]

        # save image and mask
        output_image_file = f"{output_dir}/images/image_{uuid}_{z:03d}.png"
        cv2.imwrite(output_image_file, image)
        output_mask_file = f"{output_dir}/masks/mask_{uuid}_{z:03d}.png"
        cv2.imwrite(output_mask_file, mask)

        # append list
        scan = z // 140
        circum = row[f"sweep_{scan+1}_ac_mm"]
        output_list.append([uuid, z, circum])

# write datalist
with open(output_list_file, "wt") as f:
    for out in output_list:
        f.write(f"{out[0]},{out[1]},{out[2]}\n")
