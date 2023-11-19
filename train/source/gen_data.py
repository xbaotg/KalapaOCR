from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="")
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()


data = Path(args.data)
images = list(data.glob("images/*/*.jpg"))
annotations = list(data.glob("annotations/*.txt"))
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

# shuffle
random.seed(42)
random.shuffle(annotations)

train_annotations = (root_save_dir / "train_annotations.txt").open(
    "w+", encoding="utf-8"
)
val_annotations = (root_save_dir / "val_annotations.txt").open("w+", encoding="utf-8")

for p in tqdm(annotations):
    with p.open(encoding="utf-8") as f:
        lines = f.readlines()
        random.shuffle(lines)

        for i, line in enumerate(lines):
            line = line.strip()
            image_path = line.split("\t")[0]
            folder_image = image_path.split("/")[0]

            if i < 0.9 * len(lines):
                train_annotations.write("train/" + line + "\n")

                if not (root_save_dir / "train" / folder_image).exists():
                    (root_save_dir / "train" / folder_image).mkdir(parents=True)

                shutil.copy(
                    data / "images" / image_path, root_save_dir / "train" / folder_image
                )
            else:
                val_annotations.write("val/" + line + "\n")

                if not (root_save_dir / "val" / folder_image).exists():
                    (root_save_dir / "val" / folder_image).mkdir(parents=True)

                shutil.copy(
                    data / "images" / image_path, root_save_dir / "val" / folder_image
                )

train_annotations.close()
val_annotations.close()
