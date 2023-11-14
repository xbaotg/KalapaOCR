from pathlib import Path
from tqdm import tqdm


SYNTHETIC_PATH = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/synthetic_tuongbck"


synthetic_path = Path(SYNTHETIC_PATH) / "images"
save_path = Path(SYNTHETIC_PATH) / "annotations"

for p in tqdm(synthetic_path.glob("*/*.txt")):
    par_dir = p.parent.name

    with open(p, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l.replace("  ", " ") for l in lines]
        lines = list(
            map(
                lambda l: par_dir
                + "/"
                + l.split(" ")[0]
                + "\t"
                + " ".join(l.split(" ")[1:]),
                lines,
            )
        )

        with open(save_path / (par_dir + ".txt"), "w+") as f:
            f.write("\n".join(lines))
