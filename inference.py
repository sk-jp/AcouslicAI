"""
The following is a the inference code for running the baseline algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK

from model import FetalAbdomenSegmentation

import torch

from fix_model_state_dict import fix_model_state_dict
from get_transform import get_transform
from read_yaml import read_yaml

# docker
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
# local
#INPUT_PATH = Path("./input")
#OUTPUT_PATH = Path("./output")

def run():
    print("Setting up ...")
    
    # Read a config file
#    cfg = read_yaml('flexunet_b4_test_mha.yaml')
#    cfg = read_yaml('unet_test_mha.yaml')
    cfg = read_yaml('unet_multitask_3slices_test.yaml')
    
    # Define a mode
    if cfg.Model.arch == 'flexible_unet_multitask':
        from flexible_unet_multitask import FlexibleUNetMultitask
        model = FlexibleUNetMultitask(**cfg.Model.params)  
    elif cfg.Model.arch == 'unet_multitask':
        from unet_multitask import UnetMultitask
        model = UnetMultitask(**cfg.Model.params)  
       
    # Load pretrained model weights
    print(f'  Loading: {cfg.Model.pretrained}')
    checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = fix_model_state_dict(state_dict)
    model.load_state_dict(state_dict)

    # transform
    transform = get_transform(cfg.Transform.test)

    # Read the input
    stacked_fetal_ultrasound_path = get_image_file_path(
        location=INPUT_PATH / "images/stacked-fetal-ultrasound")

    # Process the inputs: any way you'd like
#    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)

    # Instantiate the algorithm
    algorithm = FetalAbdomenSegmentation(
        model, transform, cfg)

    # Forward pass
    fetal_abdomen_segmentation, fetal_abdomen_frame_number = algorithm.predict(
        stacked_fetal_ultrasound_path)
    
    if cfg.Model.arch == 'unet_multitask':
#        print("pre:", fetal_abdomen_segmentation.shape)
        # post transform
        post_transform = get_transform(cfg.Transform.test_post)
        fetal_abdomen_segmentation = post_transform(np.expand_dims(fetal_abdomen_segmentation, 0))
        fetal_abdomen_segmentation = fetal_abdomen_segmentation[0].numpy()
#        print("post:", fetal_abdomen_segmentation.shape)

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/fetal-abdomen-segmentation",
        array=fetal_abdomen_segmentation,
        frame_number=fetal_abdomen_frame_number,
    )
    write_json_file(
        location=OUTPUT_PATH / "fetal-abdomen-frame-number.json",
        content=fetal_abdomen_frame_number
    )

    # Print the output
    print("output folder contents:")
    print_directory_contents(OUTPUT_PATH)

    # Print shape and type of the output
    print("\nprinting output shape and type:")
    print(f"  shape: {fetal_abdomen_segmentation.shape}")
    print(f"  type: {type(fetal_abdomen_segmentation)}")
    print(f"  dtype: {fetal_abdomen_segmentation.dtype}")
    print(f"  unique values: {np.unique(fetal_abdomen_segmentation)}")
    print(f"  frame number: {fetal_abdomen_frame_number}")
    print(f"  {type(fetal_abdomen_frame_number)}")

    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + \
        glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


# Get image file path from input folder
def get_image_file_path(*, location):
    input_files = glob(str(location / "*.tiff")) + \
        glob(str(location / "*.mha"))
        
    return input_files[0]


def write_array_as_image_file(*, location, array, frame_number=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    # Assert that the array is 2D
    assert array.ndim == 2, f"Expected a 2D array, got {array.ndim}D."
    
    # Convert the 2D mask to a 3D mask (this is solely for visualization purposes)
    array = convert_2d_mask_to_3d(
        mask_2d=array,
        frame_number=frame_number,
        number_of_frames=840,
    )

    image = SimpleITK.GetImageFromArray(array)
    # Set the spacing to 0.28mm in all directions
    image.SetSpacing([0.28, 0.28, 0.28])
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def convert_2d_mask_to_3d(*, mask_2d, frame_number, number_of_frames):
    # Convert a 2D mask to a 3D mask
    mask_3d = np.zeros((number_of_frames, *mask_2d.shape), dtype=np.uint8)
    # If frame_number == -1, return a 3D mask with all zeros
    if frame_number == -1:
        return mask_3d
    # If frame_number is within the valid range, set the corresponding frame to the 2D mask
    if frame_number is not None and 0 <= frame_number < number_of_frames:
        mask_3d[frame_number, :, :] = mask_2d
        return mask_3d
    # If frame_number is None or out of bounds, raise a ValueError
    else:
        raise ValueError(
            f"frame_number must be between -1 and {number_of_frames - 1}, got {frame_number}."
        )


def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print_directory_contents(child_path)
        else:
            print("  " + child_path)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
