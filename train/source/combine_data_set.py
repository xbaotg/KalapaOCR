from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="+", type=str, default=[])
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()

if args.output == "":
    print("Please specify output path")
    exit()

root_save_dir = Path(args.output)
allowd_chars = set(
    "".join(
        [
            s.strip()
            for s in open(
                "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/labels.txt"
            ).readlines()
        ]
    )
    + " "
)

if root_save_dir.exists():
    shutil.rmtree(root_save_dir)

root_save_dir.mkdir(parents=True)
(root_save_dir / "train").mkdir()
(root_save_dir / "val").mkdir()


train_annotations = (root_save_dir / "train_annotations.txt").open(
    "w+", encoding="utf-8"
)
val_annotations = (root_save_dir / "val_annotations.txt").open("w+", encoding="utf-8")

for data_id, path in enumerate(args.data):
    data = Path(path)
    images = list(data.glob("images/*/*.jpg"))
    annotations = list(data.glob("annotations/*.txt"))

    # shuffle
    random.shuffle(annotations)

    for p in tqdm(annotations):
        with p.open(encoding="utf-8") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                image_path = line.split("\t")
                original_image_path = image_path[0]
                folder_image = image_path[0].split("/")[0] + "_" + str(data_id)
                image_path[0] = (
                    folder_image + "/" + "/".join(image_path[0].split("/")[1:])
                )

                if i < 0.9 * len(lines):
                    train_annotations.write("train/" + "\t".join(image_path) + "\n")

                    if not (root_save_dir / "train" / folder_image).exists():
                        (root_save_dir / "train" / folder_image).mkdir(parents=True)

                    shutil.copy(
                        data / "images" / original_image_path,
                        root_save_dir / "train" / folder_image,
                    )
                else:
                    val_annotations.write("val/" + "\t".join(image_path) + "\n")

                    if not (root_save_dir / "val" / folder_image).exists():
                        (root_save_dir / "val" / folder_image).mkdir(parents=True)

                    shutil.copy(
                        data / "images" / original_image_path,
                        root_save_dir / "val" / folder_image,
                    )

train_annotations.close()
val_annotations.close()
