import time
from natsort import natsorted
import numpy as np
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
parser.add_argument("--weights1", type=str)
parser.add_argument("--weights2", type=str)
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
config["weights"] = args.weights1
detector1 = Predictor(config)
config["weights"] = args.weights2
detector2 = Predictor(config)
# print(trainer.model)
# torch.save(trainer.model.state_dict(), "vgg19_cutlastblock.pth")
# replace backbone to linear model
# detector.model.cnn.model = MobileNetV3(256, True, dropout=0.5)


result_file.write("id,answer\n")
t = 0

for i, p in tqdm(enumerate(natsorted(test_path.glob("*/*.jpg")))):
    print(p)
    final = None
    start_time = time.time()
    img = Image.open(p)
    s , score= detector.predict(img,True)
    s1, score1 = detector1.predict(img, True)
    s2, score2 = detector2.predict(img, True)
    tt = time.time() - start_time
    t += tt
    # print("Time: ", tt, "\t", "Mean: ", t / (i + 1))
    # print("model: ", score)
    # print("model1 ", score1)
    sum_score = np.mean(score[0])
    sum_score1 = np.mean(score1[0])
    sum_score2 = np.mean(score2[0])
    new_score = []

    
    # fs = ""
    # fs1 = ""
    # fscore = score[0][1:-1]
    # fscore1 = score1[0][1:-1]
    
    # for j in range(len(fscore)):
    #     if fscore[j] >= 0.2:
    #         fs += s[j]

    # for j in range(len(fscore1)):
    #     if fscore1[j] >= 0.2:
    #         fs1 += s1[j]

    # s = fs
    # s1 = fs1
    
    
    # if len(s) != len(s1):
    if sum_score < 0.65 or sum_score2 < 0.65 or sum_score1 < 0.65:
        s = ""
        s1 = ""
        s2 = ""
    max_score = max([sum_score,sum_score1,sum_score2])
    if max_score == sum_score:
        final = s
    elif max_score ==sum_score1:
        final =s1
    else:
        final = s2
    # else:
    #     final = []
    #     for idx in range(len(s)):
    #         if s[idx] == s1[idx]:
    #             final.append(s[idx])
    #             continue
    #         if score[0][idx] > score1[0][idx]:
    #             final.append(s[idx])
    #             new_score.append(score[0][idx])
    #         else:
    #             final.append(s1[idx])
    #             new_score.append(score1[0][idx])
    #     sum_new_score = np.mean(new_score)
    #     if sum_new_score > sum_score and sum_new_score > sum_score1:
    #         final = "".join(final)
    #         print("From both model :")
    #     elif sum_score > sum_score1:
    #         final = s
    #         print("From model :")
    #     else:
    #         final = s1
    #         print("From model1 :")
    print(final)
    result_file.write("/".join(p.parts[-2:]) + "," + final + "\n")

result_file.close()
