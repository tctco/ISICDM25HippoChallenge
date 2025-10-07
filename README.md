# ISICDM25 multi-site Hippocampus segmentation challenge

## Prerequisites

System info: 
- Ubuntu 24.04.3 LTS
- AMD Ryzen 9 7950X3D
- 64GB RAM
- NVIDIA 4090D 24GB

install [fastsurfur](https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#docker) in docker

```bash
docker pull deepmi/fastsurfer:latest

# you may need to add the current user to the docker group
sudo usermod -aG docker $USER
```

clone this repo:

```bash
git clone --recurse-submodules https://github.com/tctco/ISICDM25HippoChallenge.git
```


sync env

```bash
uv sync
```

install [TriALS](https://github.com/tctco/TriALS)
```bash
# you likely don't need to install torch, as uv sync has already handled this for you
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd TriALS
uv pip install -e .
```

## Prepare the ISICDM25 dataset

unzip the data to `project/path/ISICDM_dataset`.

> SiteA images should be renamed with extension `.nii` first.
> 
> SiteA contains some mis-labeled images: s1_label34, s1_label64, s1_label84

## Run the segmentation pipeline

```bash
sh pipeline.sh
```

## Test result

- Dice: 0.9065
- HD95: 1.9499


<p align="center"><img src="https://github.com/tctco/DCCCSlicer/blob/master/demo/dept_logo.png" style="width:15%;" /></p>
