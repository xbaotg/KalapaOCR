from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from pprint import pprint
from backbone import MobileNetV3, EfficientNet
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--data_root", type=str)
parser.add_argument("--total_iter", type=int, default=20000)
parser.add_argument("--valid_every", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--height", type=int, default=32)
parser.add_argument("--max_width", type=int, default=512)
parser.add_argument("--min_width", type=int, default=32)
parser.add_argument("--arch", type=str, default="vgg_seq2seq")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--dataset-name", type=str, default="hw")
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--base_temperature", type=float, default=0.35)
parser.add_argument("--default-loss", action="store_true", default=False)
parser.add_argument("--use-loss-best", action="store_true", default=False)
parser.add_argument("--max-lr", type=float, default=0.01)
parser.add_argument("--labels", type=str, default="")
args = parser.parse_args()


# print args
for k, v in vars(args).items():
    print(f"{k}: {v}")

vocab = (
    "".join(
        [
            s.strip()
            for s in open(
                "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/labels.txt"
            ).readlines()
        ]
    )
    + " " + str(args.labels)
)

# vocab = "nhTgiHuaàNPoBLĐCcưâpêôráơmMKìVồQạASịảyúGXộốDệ5794ậ861ĩ3ắ/Yítờ2ếẤãổấăớầỹọứóềợkòũễRùeừỳụủẩằỷdýẵEõỵèbữểỏựlửUOÂxÔđởIặvÝé'ỗÁsJFẻqẫẹẽfƯỨzẢẳỉỔỡWỞÓÍ "

print("Vocab size:", len(vocab))
input("Checking config. Press Enter to continue")


if args.model not in ["mobile", "efficient", "vgg"]:
    raise ValueError("Invalid model name")

if args.valid_every == -1:
    args.valid_every = args.total_iter

# build model
config = Cfg.load_config_from_name(args.arch)
dataset_params = {
    "name": args.dataset_name,
    "data_root": args.data_root,
    "train_annotation": "train_annotations.txt",
    "valid_annotation": "val_annotations.txt",
    "image_height": args.height,
    "image_max_width": args.max_width,
    "image_min_width": args.min_width,
    "temperature": args.temperature,
    "base_temperature": args.base_temperature,
}

params = {
    "print_every": args.log_every,
    "valid_every": args.valid_every,
    "iters": args.total_iter,
    "checkpoint": args.checkpoint,
    "export": args.output,
    "metrics": 500000,
    "batch_size": args.bs,
}

config['dataloader']['num_workers'] = 3
config["pretrain"] = False
config["weights"] = ""
config["trainer"].update(params)
config["dataset"].update(dataset_params)
config["device"] = "cuda:0"
config['transformer']['dropout'] = 0.1
config["optimizer"]["max_lr"] = args.max_lr
config["vocab"] = vocab

pprint(config)

# build trainer
trainer = Trainer(config, pretrained=False, default_loss=args.default_loss, use_loss_best=args.use_loss_best)

if args.model == "efficient":
    trainer.model.cnn.model = EfficientNet(256, True, dropout=0.1)
    print(trainer.model)
    input()
elif args.model == "mobile":
    trainer.model.cnn.model = MobileNetV3(256, True, dropout=0.1)
    print(trainer.model)
    input()
else:
    print(trainer.model)

trainer.model.to("cuda")

# trainer.config.save("./config.yml")
if args.checkpoint:
    print("Load checkpoint from", args.checkpoint)
    trainer.load_weights(args.checkpoint)


# train
print("Start training")
trainer.train()

if args.valid_every == args.total_iter:
    trainer.save_weights(args.output.replace(".pt", "_done.pt"))
elif not Path(args.output).exists():
    trainer.save_weights(args.output.replace(".pt", "_done.pt"))
