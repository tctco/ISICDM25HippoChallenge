import os, subprocess
from argparse import ArgumentParser

env = os.environ.copy()
FILE_PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
env["nnUNet_raw"] = os.path.join(FILE_PARENT_PATH, "nnUNet_folders", "nnUNet_raw")
env["nnUNet_preprocessed"] = os.path.join(
    FILE_PARENT_PATH, "nnUNet_folders", "nnUNet_preprocessed"
)
env["nnUNet_results"] = os.path.join(
    FILE_PARENT_PATH, "nnUNet_folders", "nnUNet_results"
)


def prepare_nnunet_dataset(dataset_id):
    # 依次跑命令（示例）
    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "-pl",
            "ResEncUNetPlanner",
            "-gpu_memory_target",
            "24",
            "--verify_dataset_integrity",
        ],
        env=env,
        check=True,
    )


def train_nnunet_model(dataset_id):
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            "0",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-p",
            "nnUNetResEncUNetPlans",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            "1",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-p",
            "nnUNetResEncUNetPlans",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            "2",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-p",
            "nnUNetResEncUNetPlans",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            "3",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-p",
            "nnUNetResEncUNetPlans",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            "4",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-p",
            "nnUNetResEncUNetPlans",
        ],
        env=env,
        check=True,
    )


def prepare_mednext_dataset(dataset_id):
    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "-gpu_memory_target",
            "24",
            "--verify_dataset_integrity",
        ],
        env=env,
        check=True,
    )


def train_mednext(dataset_id, demo=False):
    trainer = (
        "nnUNetTrainerV2_MedNeXt_B_kernel5_1epochs"
        if demo
        else "nnUNetTrainerV2_MedNeXt_B_kernel5_100epochs"
    )
    for i in range(5):
        subprocess.run(
            [
                "nnUNetv2_train",
                str(dataset_id),
                "3d_fullres",
                str(i),
                "-tr",
                trainer,
            ],
            env=env,
            check=True,
        )


def predict_nnunet_model(dataset_id):
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i",
            os.path.join(
                FILE_PARENT_PATH,
                "nnUNet_folders",
                "nnUNet_raw",
                "Dataset" + str(dataset_id) + "_HippoROI",
                "imagesTs",
            ),
            "-o",
            os.path.join(FILE_PARENT_PATH, "submit", "nnunet_results"),
            "-d",
            str(dataset_id),
            "-c",
            "3d_fullres",
            "-p",
            "nnUNetResEncUNetPlans",
            "-tr",
            "nnUNetTrainer_200epochs",
            "-chk",
            "checkpoint_best.pth",
        ],
        env=env,
        check=True,
    )


def predict_mednext(dataset_id, demo=False):
    trainer = "nnUNetTrainerV2_MedNeXt_B_kernel5_1epochs" if demo else "nnUNetTrainerV2_MedNeXt_B_kernel5_100epochs"
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i",
            os.path.join(
                FILE_PARENT_PATH,
                "nnUNet_folders",
                "nnUNet_raw",
                "Dataset" + str(dataset_id) + "_HippoROI",
                "imagesTs",
            ),
            "-o",
            os.path.join(FILE_PARENT_PATH, "submit", "mednext_results"),
            "-d",
            str(dataset_id),
            "-c",
            "3d_fullres",
            "-tr",
            trainer,
            "-chk",
            "checkpoint_best.pth",
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--train_mednext", action="store_true")
    # parser.add_argument("--prepare_mednext", action="store_true")
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--predict_mednext", action="store_true")
    parser.add_argument("--demo", action="store_true", default=False)
    args = parser.parse_args()
    if args.prepare:
        prepare_nnunet_dataset(args.dataset_id)
    if args.train:
        train_nnunet_model(args.dataset_id)
    if args.predict:
        predict_nnunet_model(args.dataset_id)
    if args.train_mednext:
        train_mednext(args.dataset_id, args.demo)
    # if args.prepare_mednext:
    #     prepare_mednext_dataset(args.dataset_id)
    if args.predict_mednext:
        predict_mednext(args.dataset_id, args.demo)
