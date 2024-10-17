import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

slice_num = 0

file_path = "/home/jovyan/shared/xinzeli/CORF/data/ACDC/data/"

all_image_path = sorted(glob.glob(file_path + "/**/*.nii.gz", recursive=True))

mask_path = sorted(glob.glob(file_path + "/**/*_gt.nii.gz", recursive=True))

image_path = [image for image in all_image_path if "4d" not in image and "gt" not in image]

def process_images(image_path, mask_path, output_dir):
    slice_num = 0
    numberOfImages = len(image_path)

    for i in range(numberOfImages):
        try:
            image_file = image_path[i]
            mask_file = mask_path[i]
            img_itk = sitk.ReadImage(image_file)
            msk_itk = sitk.ReadImage(mask_file)
            
            image = sitk.GetArrayFromImage(img_itk)
            mask = sitk.GetArrayFromImage(msk_itk)

            if image.shape != mask.shape:
                print(f"Dimension mismatch in {image_file}")
                continue

            image = (image - image.min()) / (image.max() - image.min()).astype(np.float32)

            for slice_ind in range(image.shape[0]):
                with h5py.File(os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_slice_{slice_ind}.h5'), 'w') as f:
                    f.create_dataset('image', data=image[slice_ind], compression="gzip")
                    f.create_dataset('label', data=mask[slice_ind], compression="gzip")
                    slice_num += 1

        except Exception as e:
            print(f"Failed processing {image_file}: {str(e)}")

    print("Converted all ACDC volumes to 2D slices")
    print(f"Total {slice_num} slices processed")

file_path = "/home/jovyan/shared/xinzeli/CORF/data/ACDC/data/"
output_dir = "/home/jovyan/shared/xinzeli/CORF/data/ACDC/training_set"
all_image_path = sorted(glob.glob(file_path + "/**/*.nii.gz", recursive=True))
mask_path = sorted(glob.glob(file_path + "/**/*_gt.nii.gz", recursive=True))
image_path = [image for image in all_image_path if "4d" not in image and "gt" not in image]
process_images(image_path, mask_path, output_dir)