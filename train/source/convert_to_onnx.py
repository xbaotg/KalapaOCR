import torch
import argparse
from pathlib import Path
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
args = parser.parse_args()

labels = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/labels.txt"
result_file = Path(args.result_file).open("w+")

config = Cfg.load_config_from_name("vgg_seq2seq")
config["weights"] = args.weights
config["cnn"]["pretrained"] = False
config["device"] = "cpu"
vocab = "".join([s.strip() for s in open(labels, encoding="utf-8").readlines()]) + " "
config["vocab"] = vocab
dataset_params = {
    "image_height": args.height,
    "image_max_width": args.max_width,
    "image_min_width": args.min_width,
    "data_root": "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/train/VietOCR/data_all",
    "name": "all",
    "train_annotation": "train_annotations.txt",
    "valid_annotation": "val_annotations.txt",
}
config["dataset"].update(dataset_params)
config['trainer']['checkpoint'] = args.weights
# detector = Predictor(config)

from vietocr.tool.predictor import Predictor

trainer = Predictor(config)
trainer.model.eval()

# replace backbone to linear model
# detector.model.cnn.model = EfficientNet(256, True, dropout=0.5)
# from efficientnet_pytorch import EfficientNet
# detector.model.cnn.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=256)
# print(detector.model)

# pytorch_total_params = sum(p.numel() for p in detector.model.cnn.parameters())
# print(pytorch_total_params)
# torch.save(detector.model, "mobilenetv3.pth")


def convert_cnn_part(img, save_path, model):
    with torch.no_grad():
        src = model.cnn(img)
        torch.onnx.export(
            model.cnn,
            img,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            verbose=True,
            input_names=["img"],
            output_names=["output"],
            dynamic_axes={"img": {3: "lenght"}, "output": {0: "channel"}},
        )

    return src


def convert_encoder_part(model, src, save_path):
    encoder_outputs, hidden = model.transformer.encoder(src)
    torch.onnx.export(
        model.transformer.encoder,
        src,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["src"],
        output_names=["encoder_outputs", "hidden"],
        dynamic_axes={
            "src": {0: "channel_input"},
            "encoder_outputs": {0: "channel_output"},
        },
    )
    return hidden, encoder_outputs


def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
    tgt = tgt[-1]

    torch.onnx.export(
        model.transformer.decoder,
        (tgt, hidden, encoder_outputs),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["tgt", "hidden", "encoder_outputs"],
        output_names=["output", "hidden_out", "last"],
        dynamic_axes={
            "encoder_outputs": {0: "channel_input"},
            "last": {0: "channel_output"},
        },
    )


img = torch.rand(1, 3, 32, 512)
src = convert_cnn_part(img, "./weights/cnn.onnx", trainer.model)
hidden, encoder_outputs = convert_encoder_part(
    trainer.model, src, "./weights/encoder.onnx"
)
device = img.device
tgt = torch.LongTensor([[1] * len(img)]).to(device)
convert_decoder_part(
    trainer.model, tgt, hidden, encoder_outputs, "./weights/decoder.onnx"
)
