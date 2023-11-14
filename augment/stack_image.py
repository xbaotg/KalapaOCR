import numpy as np
import os
import glob
from PIL import Image, ImageDraw
from tqdm import tqdm
def get_int_path(path):
    return int(os.path.basename(path).split(".")[0])

dir_path = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/public/images"
output_path = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/stacked"
folder_path = glob.glob(os.path.join(dir_path, "*"))
for folder in tqdm(folder_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imgs_path = sorted(glob.glob(os.path.join(folder, "*")),key=get_int_path)

    img_list = []
    final_size = [0,0]
    for img in imgs_path:
        img = Image.open(img)
        w,h = img.size
        final_size[0] = w
        final_size[1] += h
        img_list.append(img)
    final_image = Image.new("RGB", (final_size[0], final_size[1]))
    y = 0
    for img in img_list:
        w,h = img.size
        final_image.paste(img, (0,y))
        y += h
    # final_path = os.path.join(output_path, os.path.basename(folder)+".jpg")
    final_image.save(os.path.join(output_path, os.path.basename(folder)+".jpg"))