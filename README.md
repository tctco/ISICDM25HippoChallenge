# ISICDM25 multi-site Hippocampus segmentation challenge

## prerequisites

install [fastsurfur](https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#docker) in docker

```bash
docker pull deepmi/fastsurfer:latest

# you may need to add the current user to the docker group
sudo usermod -aG docker $USER
```

install [TriALS](https://github.com/tctco/TriALS)
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd TriALS
uv pip install -e .
```

sync env

```bash
uv sync
```

## prepare the ISICDM25 dataset

unzip the data to `project/path/ISICDM_dataset`.

> SiteA images should be renamed with extension `.nii` first.

## run the segmentation pipeline

```bash
sh pipeline.sh
```

## test result

- Dice: 0.9065
- HD95: 1.9499