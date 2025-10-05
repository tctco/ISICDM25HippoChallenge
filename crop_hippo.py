'''
Crop the hippocampus from the fastsurfur preprocessed data.
'''

from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
from monai.transforms.utils import generate_spatial_bounding_box
from monai.transforms import SpatialCrop, SaveImage, LoadImage, ResampleToMatch
from monai.data import MetaTensor
from tqdm import tqdm
from collections import defaultdict

from monai.visualize.utils import blend_images
import matplotlib.pyplot as plt
from monai.data import MetaTensor
import nibabel as nib

loader = LoadImage(ensure_channel_first=True)

mask_resampler = ResampleToMatch(
    mode='nearest'
)

def show_blend_image(orig:MetaTensor, seg:MetaTensor, slice_index:int=120, axis:int=0):

    slicer = [slice(None)] * orig.ndim
    slicer[axis] = slice_index
    slicer = tuple(slicer)  # 转成元组切片

    image = orig.permute(2, 1, 3, 0)[slicer]
    blended = blend_images(orig, seg)
    blended = blended.permute(2, 1, 3, 0)[slicer]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap="gray")
    axs[0].title.set_text("Image")
    axs[1].imshow(blended)
    axs[1].title.set_text("Image with overlaid label")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    plt.suptitle("Image and label visualization")
    plt.tight_layout()
    plt.show()

# ------------ 工具函数 ------------
def _to_numpy(x):
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def spacing_from_affine(aff):
    aff = np.asarray(aff)
    return np.sqrt((aff[:3, :3] ** 2).sum(0))  # (sx, sy, sz)

def bbox_from_labels(seg, label_ids, margin=(0, 0, 0)):
    """
    seg: [H,W,D] (Tensor/MetaTensor/ndarray), 离散标签; label_ids: 要取前景的标签列表
    返回: dict(start,end,center,size,size_mm)
    """
    arr = _to_numpy(seg)
    fg = np.isin(arr, np.asarray(label_ids))
    if not fg.any():
        return None
    start, end = generate_spatial_bounding_box(
        fg.astype(np.uint8), select_fn=lambda x: x > 0, margin=list(margin)
    )  # end 为"排他"索引
    start = np.array(start, dtype=int)
    end   = np.array(end,   dtype=int)
    size  = end - start                      # (dz,dw,dd) in voxels
    center = np.round((start + end - 1) / 2.).astype(int)
    # 取 spacing：优先 seg.meta 里的 affine；否则返回 None
    aff = None
    if isinstance(seg, MetaTensor):
        aff = seg.affine
    size_mm = None
    if aff is not None:
        sp = spacing_from_affine(aff)
        size_mm = size * sp
    return dict(start=start, end=end, center=center, size=size, size_mm=size_mm)

def crop_by_center(orig, center, roi_size):
    """
    用中心+尺寸裁剪。orig 可为 MetaTensor/张量。返回与输入同类型(若为MetaTensor会带meta)。
    roi_size: (z,y,x) int
    """
    # cropped_data = orig[0, center[0] - roi_size[0] // 2:center[0] + roi_size[0] // 2,
    #                     center[1] - roi_size[1] // 2:center[1] + roi_size[1] // 2,
    #                     center[2] - roi_size[2] // 2:center[2] + roi_size[2] // 2]
    cropper = SpatialCrop(roi_center=tuple(center.tolist()), roi_size=tuple(roi_size.tolist()))
    result = cropper(orig)
    # result[:] = cropped_data.as_tensor()
    return result

def save_nii(img, out_path, output_postfix=None):
    """
    用 MONAI SaveImage 保存为 NIfTI（会从 MetaTensor 读取 affine/meta）。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saver = SaveImage(
        output_dir=out_path.parent,
        output_postfix=output_postfix or "",
        output_ext=".nii.gz",
        separate_folder=False,
        print_log=False,
    )
    saver(img, meta_data=getattr(img, "meta", None), filename=out_path)

# ------------ 单例处理：拿到左右海马中心与 bbox，并各自裁剪保存 ------------
def process_one_case(
    orig: MetaTensor, 
    seg: MetaTensor, 
    out_dir: Path, 
    margin=(4, 4, 4), 
    use_mean_roi_size=None, 
    prefix='img',
    return_cropped_seg=False):
    """
    orig: 原图 (MetaTensor/Tensor)   seg: 对齐到 orig 的分割
    margin: bbox 外扩 (voxel)
    use_mean_roi_size: 若给定 (z,y,x) 将用该固定尺寸按中心裁剪；否则用每例自身 bbox size
    """
    results = defaultdict(dict)
    out_dir = Path(out_dir)
    for name, lab in [("left", [17]), ("right", [53])]:
        bbox = bbox_from_labels(seg, lab, margin=margin)
        if bbox is None:
            print(f"[WARN] {name} hippocampus not found, skip.")
            continue
        center = bbox["center"]
        roi_size = bbox["size"] if use_mean_roi_size is None else np.array(use_mean_roi_size, dtype=int)
        # 防越界保护：裁剪时 SpatialCrop 要求 ROI 完全在图内
        roi_size = np.minimum(roi_size, np.array(orig.shape[-3:], dtype=int))
        cropped_img = crop_by_center(orig, center, roi_size)
        seg_copy = seg.clone()
        seg_copy[~torch.isin(seg_copy, torch.tensor(lab))] = 0
        seg_copy[torch.isin(seg_copy, torch.tensor(lab))] = 1
        if prefix == 'gt':
            orig[orig > 0] = 1
        cropped_seg = crop_by_center(seg_copy,  center, roi_size)
        # 保存
        im_out = out_dir / f"{name}_hippo_{prefix}"
        sg_out = out_dir / f"{name}_hippo_seg"
        save_nii(cropped_img, im_out)
        save_nii(cropped_seg, sg_out)
        results[name] = dict(center_vox=center, roi_size_vox=roi_size, bbox=bbox)
    if return_cropped_seg:
        results['orig'] = orig
    return results

# ------------ 统计整个数据集的“均值长宽高” ------------
def collect_mean_box_sizes(seg_paths, margin=(4,4,4), label_list=(("left", [17]), ("right", [53]))):
    """
    seg_paths: 可迭代的分割路径列表（已与各自原图空间对齐）
    返回: {"left": mean_size_vox, "right": mean_size_vox}
    """
    acc = {"left": [], "right": []}
    result = []
    for p in tqdm(seg_paths):
        try:
            seg = loader(p)
        except Exception as e:
            print('failed:', p, e, sep='\n')
            continue

        for name, lab in label_list:
            b = bbox_from_labels(seg, lab, margin=margin)
            if b is not None:
                acc[name].append(b["size"])
                result.append(b)
    means = {}
    for k, arrs in acc.items():
        if len(arrs):
            means[k] = np.round(np.mean(np.stack(arrs, 0), 0)).astype(int)
    return means, acc, result


def convert_mgz_to_nii(preprocessed_root:Path):
    for site_dir in preprocessed_root.iterdir():
        for img_dir in tqdm(list(site_dir.iterdir())):
            seg_nii_path = img_dir / 'mri' / 'aparc.DKTatlas+aseg.deep.nii.gz'
            if not seg_nii_path.exists():
                seg_mgz_path = img_dir / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
                seg = nib.load(seg_mgz_path)
                seg_nii_path = img_dir / 'mri' / 'aparc.DKTatlas+aseg.deep.nii.gz'
                nib.save(seg, seg_nii_path)

            orig_nii_path = img_dir / 'mri' / 'orig' / '001.nii.gz'
            if not orig_nii_path.exists():
                orig_mgz_path = img_dir / 'mri' / 'orig' / '001.mgz'
                orig = nib.load(orig_mgz_path)
                nib.save(orig, orig_nii_path)

def crop_hippo(preprocessed_root:Path, gt_root: Path | None = None):  
    bbox_size = (48, 64, 64)
    for site_dir in preprocessed_root.iterdir():
        if gt_root:
            site_gt_root = gt_root / site_dir.name / 'label'
        for img_dir in tqdm(list(site_dir.iterdir())):
            orig = loader(img_dir / 'mri' / 'orig' / '001.nii.gz')
            seg = loader(img_dir / 'mri' / 'aparc.DKTatlas+aseg.deep.nii.gz')
            seg = mask_resampler(seg, orig)
            results = process_one_case(
                orig, seg, out_dir=img_dir / 'mri', 
                margin=(0,0,0), use_mean_roi_size=bbox_size, 
                prefix='img')
            if gt_root:
                gt_path = site_gt_root / f'{img_dir.name.replace("image", "label")}.nii'
                if not gt_path.exists():
                    gt_path = gt_path.with_suffix('.nii.gz')
                if not gt_path.exists():
                    raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
                gt = loader(gt_path)
                results = process_one_case(
                    gt, seg, out_dir=img_dir / 'mri', 
                    margin=(0,0,0), use_mean_roi_size=bbox_size, 
                    prefix='gt', return_cropped_seg=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fastsurfur_preprocessed_path', type=str, required=True)
    parser.add_argument('--gt_root', type=str, required=False)
    args = parser.parse_args()
    fastsurfur_preprocessed_path = args.fastsurfur_preprocessed_path
    gt_root = args.gt_root
    if gt_root:
        gt_root = Path(gt_root)
    else:
        gt_root = None
    preprocessed_root = Path(fastsurfur_preprocessed_path)
    convert_mgz_to_nii(preprocessed_root)
    crop_hippo(preprocessed_root, gt_root)


