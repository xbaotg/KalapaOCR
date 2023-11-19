import onnx
import onnxruntime
import numpy as np
import torch
import time
import math
from pathlib import Path
from natsort import natsorted
from vietocr.model.vocab import Vocab
from torch.nn.functional import log_softmax, softmax
from tqdm import tqdm
from PIL import Image


def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session

    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)  # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token] * len(img)]
    char_probs = [[1] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {
            decoder_session.get_inputs()[0].name: tgt_inp[-1],
            decoder_session.get_inputs()[1].name: hidden,
            decoder_session.get_inputs()[2].name: encoder_outputs,
        }

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)
        output = softmax(output, dim=-1)

        values, indices = torch.topk(output, 5)

        indices = indices[:, -1, 0]
        indices = indices.tolist()

        values = values[:, -1, 0]
        values = values.tolist()
        char_probs.append(values)

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T
    char_probs = np.asarray(char_probs).T
    char_probs = np.multiply(char_probs, translated_sentence > 3)

    return translated_sentence, char_probs


# create inference session
cnn_session = onnxruntime.InferenceSession("./weights/cnn.onnx")
encoder_session = onnxruntime.InferenceSession("./weights/encoder.onnx")
decoder_session = onnxruntime.InferenceSession("./weights/decoder.onnx")
vocab = Vocab(
    "nhTgiHuaàNPoBLĐCcưâpêôráơmMKìVồQạASịảyúGXộốDệ5794ậ861ĩ3ắ/Yítờ2ếẤãổấăớầỹọứóềợkòũễRùeừỳụủẩằỷdýẵEõỵèbữểỏựlửUOÂxÔđởIặvÝé'ỗÁsJFẻqẫẹẽfƯỨzẢẳỉỔỡWỞÓÍ "
)

session = (cnn_session, encoder_session, decoder_session)


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert("RGB")

    w, h = img.size
    # new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    new_w = image_max_width
    img = img.resize((new_w, image_height), Image.LANCZOS)
    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = img.astype(np.float32)
    return img


test_path = Path("/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/public/images")
result_file = Path("results_infer_onnx.txt").open("w+")
t = 0

for i, p in tqdm(enumerate(natsorted(test_path.glob("*/*.jpg")))):
    start_time = time.time()
    img = Image.open(p)
    img = process_input(img, 32, 32, 512)
    s, prob = translate_onnx(img, session)
    s = s[0].tolist()
    s = vocab.decode(s)
    tt = time.time() - start_time
    t += tt
    print("Time: ", tt, "\t", "Mean: ", t / (i + 1), "\t", s)
    result_file.write("/".join(p.parts[-2:]) + "," + s + "\n")

result_file.close()