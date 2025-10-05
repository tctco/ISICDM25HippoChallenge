from pathlib import Path
import subprocess
import os
from loguru import logger
from argparse import ArgumentParser

logger.add("./fastsurfur_preprocess.log", rotation="10 MB", retention="10 days")
project_root = Path(__file__).parent

def fastsurfur_preprocess(data_root:Path, output_root:Path):
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    for site in ("site1", "site2"):
        site_output = output_root / site
        site_output.mkdir(parents=True, exist_ok=True)
        image_dir = data_root / site / "image"
        files = []
        for pattern in ("*.nii", "*.nii.gz"):
            files.extend(sorted(image_dir.glob(pattern)))
        for path in files:
            subject = path.name.split(".nii")[0]
            cmd = [
                "sudo",
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--user",
                uid_gid,
                "-v",
                f"{data_root}:/data:ro",
                "-v",
                f"{output_root}:/output",
                "deepmi/fastsurfer:latest",
                "--t1",
                f"/data/{site}/image/{path.name}",
                "--sid",
                subject,
                "--sd",
                f"/output/{site}",
                "--seg_only",
                "--no_cereb",
                "--no_biasfield",
                "--no_hypothal"
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"命令执行失败: {e}")
                continue
            except Exception as e:
                logger.error(f"发生异常: {e}")
                continue

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fastsurfur_preprocess(data_root, output_root)