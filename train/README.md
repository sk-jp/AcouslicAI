To train a model.
1. Decode mha files with "decode_images_masks.py".
2. Create data list files for training and validation with "make_datalist.py".
3. Edit unet_multitask_3slices.yaml.
   Especially, change the following parts depending on where the decoded images and masks are stored, and what the data list files are.
     Data.dataset.top_dir
     Data.dataset.train_datalist
     Data.dataset.valid_datalist
5. Run main.py
   $ python main.py --config unet_multitask_3slices.yaml --gpus "gpu ids"
