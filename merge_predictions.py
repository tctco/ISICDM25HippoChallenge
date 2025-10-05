from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
from tqdm import tqdm

# ORIGINAL_IMAGE_ROOT = Path('/home/cheng/project/datasets/ISICDMHippoChallenge/hippo-datasetA/')
# ORIGINAL_IMAGE_ROOT = Path('/home/cheng/project/datasets/ISICDMHippoChallenge/hippo-datasetB_img/')

def resample_cropped_seg_back_to_original(seg_cropped, orig):
    if isinstance(seg_cropped, str) or isinstance(seg_cropped, Path):
        seg_cropped = nib.load(seg_cropped)
    if isinstance(orig, str) or isinstance(orig, Path):
        orig = nib.load(orig)
    seg_cropped_resampled = resample_to_img(
        seg_cropped, 
        orig, 
        interpolation='nearest', 
        force_resample=True, 
        copy_header=True
    )
    assert seg_cropped_resampled.get_fdata().max() == 1
    assert seg_cropped_resampled.get_fdata().min() == 0
    return seg_cropped_resampled

def gather_predictions(predictions_dir):
    predictions_dir = Path(predictions_dir)
    predictions = defaultdict(list)
    for prediction in predictions_dir.glob("*.nii.gz"):
        site_id = prediction.stem.split('_')[0]
        case_id = prediction.stem.split('_')[1]
        predictions[f'{site_id}_{case_id}'].append(prediction)
    for k in predictions:
        predictions[k] = sorted(predictions[k])
    return predictions

def merge_left_right_hippo_to_one(left, right):
    if isinstance(left, str) or isinstance(left, Path):
        left = nib.load(left)
    if isinstance(right, str) or isinstance(right, Path):
        right = nib.load(right)
    left_arr = left.get_fdata()
    right_arr = right.get_fdata()
    zeros = np.zeros_like(left_arr)
    zeros[left_arr > 0] = 1
    zeros[right_arr > 0] = 2
    zeros = np.round(zeros).astype(int)
    out = nib.nifti1.Nifti1Image(zeros, left.affine, left.header)
    return out

def query_original_image(site_id, case_id, original_image_root: Path):
    original_image_path = original_image_root / f'{site_id.replace("s", "site")}/image/{site_id}_{case_id}.nii.gz'
    if original_image_path.exists():
        return nib.load(original_image_path)
    else:
        original_image_path = original_image_root / f'{site_id.replace("s", "site")}/image/{site_id}_{case_id}.nii'
        if original_image_path.exists():
            return nib.load(original_image_path)
        else:
            raise FileNotFoundError(f'Original image not found for {original_image_path}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--predictions_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--original_image_root', type=str, required=True)
    args = parser.parse_args()
    predictions_dir = Path(args.predictions_dir)
    original_image_root = Path(args.original_image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = gather_predictions(predictions_dir)
    for k in tqdm(predictions):
        left, right = predictions[k]
        site_id, case_id = k.split('_')
        original_image = query_original_image(site_id, case_id, original_image_root)
        left_in_original_space = resample_cropped_seg_back_to_original(left, original_image)
        right_in_original_space = resample_cropped_seg_back_to_original(right, original_image)
        merged = merge_left_right_hippo_to_one(left_in_original_space, right_in_original_space)
        assert merged.get_fdata().max() == 2, print(k, merged.get_fdata().max())
        assert merged.get_fdata().min() == 0, print(k, merged.get_fdata().min())
        output_root = output_dir / f'{site_id.replace("s", "site")}' / 'label' 
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f'{site_id}_{case_id.replace("image", "label")}.nii.gz'
        nib.save(merged, output_path)
