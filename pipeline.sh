sudo usermod -aG docker $USER
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
uv run fastsurfur_preprocess.py \
  --data_root "${SCRIPT_DIR}/ISICDM_dataset/hippo-datasetA" \
  --output_root "${SCRIPT_DIR}/preprocessed_hippo/hippo-datasetA" && \
uv run fastsurfur_preprocess.py \
  --data_root "${SCRIPT_DIR}/ISICDM_dataset/hippo-datasetB_img" \
  --output_root "${SCRIPT_DIR}/preprocessed_hippo/hippo-datasetB" && \
uv run crop_hippo.py \
  --fastsurfur_preprocessed_path ./preprocessed_hippo/hippo-datasetA \
  --gt_root ./ISICDM_dataset/hippo-datasetA && \
uv run crop_hippo.py \
  --fastsurfur_preprocessed_path ./preprocessed_hippo/hippo-datasetB && \
uv run make_nnunetv2_dataset.py \
  --src_tr ./preprocessed_hippo/hippo-datasetA \
  --src_ts ./preprocessed_hippo/hippo-datasetB && \
uv run nnunetv2_start.py \
  --prepare \
  --dataset_id 502 \
  --train_mednext --demo \
  --predict_mednext && \
uv run merge_predictions.py \
  --predictions_dir ./submit/mednext_results \
  --output_dir ./submit/mednext_submit \
  --original_image_root ./ISICDM_dataset/hippo-datasetB_img