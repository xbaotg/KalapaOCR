import time
from natsort import natsorted
import torch.nn as nn
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from backbone import MobileNetV3
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path",
    type=str,
    default="/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/public/images",
)
parser.add_argument("--result_file", type=str, default="./result.csv")
parser.add_argument("--weights", type=str)
parser.add_argument("--height", type=int, default=32)
parser.add_argument("--max_width", type=int, default=512)
parser.add_argument("--min_width", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--labels", type=str, default="")
args = parser.parse_args()

labels = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/labels.txt"
test_path = Path(args.test_path)
result_file = Path(args.result_file).open("w+")

config = Cfg.load_config_from_name("vgg_seq2seq")
config["weights"] = args.weights
config["cnn"]["pretrained"] = False
config["device"] = args.device
vocab = "".join([s.strip() for s in open(labels).readlines()]) + " " + args.labels
config["vocab"] = vocab
dataset_params = {
    "image_height": args.height,
    "image_max_width": args.max_width,
    "image_min_width": args.min_width,
}
config["dataset"].update(dataset_params)
# config['predictor']['backbone'] = 'vgg11_bn'

detector = Predictor(config)
# print(trainer.model)
# torch.save(trainer.model.state_dict(), "vgg19_cutlastblock.pth")
# replace backbone to linear model
# detector.model.cnn.model = MobileNetV3(256, True, dropout=0.5)


result_file.write("id,answer\n")
t = 0

for i, p in tqdm(enumerate(natsorted(test_path.glob("*/*.jpg")))):
    start_time = time.time()
    img = Image.open(p)
    s, prob = detector.predict(img, True)
    tt = time.time() - start_time
    t += tt

    final = ""
    prob = prob[0][1:-1]
    for j in range(len(prob)):
        if prob[j] >= 0.2:
            final += s[j]
    s = final

    # if any(t < 0.2 for t in prob[0][1:-1]):
    #     print(p)
    #     print(prob)
    #     print(s)
    #     idx = prob[0][1:-1].tolist().index(min(prob[0][1:-1]))
    #     print(s[idx], prob[0][idx + 1])
    #     input()

    # prob = prob[0][1:-1]

    # if prob[-1] < 0.5:
    #     print(prob)
    #     print(s)
    #     input()
    
    # print("Time: ", tt, "\t", "Mean: ", t / (i + 1))

    result_file.write("/".join(p.parts[-2:]) + "," + s + "\n")

result_file.close()
