import os, json, shutil, re, glob
from argparse import ArgumentParser
from pathlib import Path
from TriALS.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

DS_ID = 502
DS_NAME = f"Dataset{DS_ID:03d}_HippoROI"
raw_root = Path(__file__).parent / "nnUNet_folders" / "nnUNet_raw"
os.environ["nnUNet_raw"] = str(raw_root)
RAW_ROOT = os.environ["nnUNet_raw"]
OUT = os.path.join(RAW_ROOT, DS_NAME)
IMG_TR = os.path.join(OUT, "imagesTr")
LBL_TR = os.path.join(OUT, "labelsTr")
IMG_TS = os.path.join(OUT, "imagesTs")
os.makedirs(IMG_TR, exist_ok=True); os.makedirs(LBL_TR, exist_ok=True); os.makedirs(IMG_TS, exist_ok=True)

USE_FS_SEG_AS_EXTRA_CHANNEL = False  # True = 两通道输入：img + fastsurfer seg

def norm_id(s):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_")


def build_training_dataset(src_tr:str):
    pairs = []
    # --- 训练集 ---
    for mri_dir in glob.glob(os.path.join(src_tr, "site*", "s*_image*", "mri")):
        case_id = Path(mri_dir).parts[-2]
        for side in ["left", "right"]:
            img = os.path.join(mri_dir, f"{side}_hippo_img.nii.gz")
            gt  = os.path.join(mri_dir, f"{side}_hippo_gt.nii.gz")
            fs  = os.path.join(mri_dir, f"{side}_hippo_seg.nii.gz")
            if not (os.path.exists(img) and os.path.exists(gt)):
                continue
            nid = norm_id(f"{case_id}_{side}")
            if USE_FS_SEG_AS_EXTRA_CHANNEL:
                shutil.copy(img, os.path.join(IMG_TR, f"{nid}_0000.nii.gz"))
                shutil.copy(fs,  os.path.join(IMG_TR, f"{nid}_0001.nii.gz"))
            else:
                shutil.copy(img, os.path.join(IMG_TR, f"{nid}_0000.nii.gz"))
            shutil.copy(gt,  os.path.join(LBL_TR, f"{nid}.nii.gz"))
            pairs.append(nid)

    print(f"Prepared {len(pairs)} training samples at {OUT}")
    return pairs

def build_testing_dataset(src_ts:str):
    # --- 测试集 ---
    ts_cases = []
    for mri_dir in glob.glob(os.path.join(src_ts, "site*", "s*_image*", "mri")):
        case_id = Path(mri_dir).parts[-2]
        for side in ["left", "right"]:
            img = os.path.join(mri_dir, f"{side}_hippo_img.nii.gz")
            fs  = os.path.join(mri_dir, f"{side}_hippo_seg.nii.gz")
            if not os.path.exists(img):
                continue
            nid = norm_id(f"{case_id}_{side}")
            if USE_FS_SEG_AS_EXTRA_CHANNEL:
                shutil.copy(img, os.path.join(IMG_TS, f"{nid}_0000.nii.gz"))
                if os.path.exists(fs):
                    shutil.copy(fs, os.path.join(IMG_TS, f"{nid}_0001.nii.gz"))
            else:
                shutil.copy(img, os.path.join(IMG_TS, f"{nid}_0000.nii.gz"))
            ts_cases.append(nid)

    print(f"Prepared {len(ts_cases)} testing samples at {OUT}")

def _generate_dataset_json(pairs:list):
    # 生成 dataset.json
    channel_names = {0: "T1_ROI"}
    if USE_FS_SEG_AS_EXTRA_CHANNEL:
        channel_names[1] = "FS_mask"
    labels = {"background": 0, "hippocampus": 1}

    generate_dataset_json(
        output_folder=OUT,
        channel_names=channel_names,
        labels=labels,
        num_training_cases=len(pairs),
        file_ending=".nii.gz",
        dataset_name=DS_NAME,
        description="Hippocampus ROI segmentation (cropped by FastSurfer)",
        reference="ISICDM Hippo Challenge",
        overwrite_json=True,
    )
    print("dataset.json written.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_tr", type=str, required=True)
    parser.add_argument("--src_ts", type=str, required=True)
    args = parser.parse_args()
    pairs = build_training_dataset(args.src_tr)
    _generate_dataset_json(pairs)
    build_testing_dataset(args.src_ts)