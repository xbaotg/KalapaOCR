from tacobox import Taco
import cv2 as cv
import os
from tqdm import tqdm

dir_path = r"/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/base/train/images"
# output_path = r"/mlcv1/WorkingSpace/Personal/baotg/Kalapa/argument_data/0.jpg"
# input_img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)


mytaco_ver = Taco(cp_vertical=0.01, max_tw_vertical=3, min_tw_vertical=1)
mytaco_hor = Taco(cp_horizontal=0.005, max_tw_horizontal=3, min_tw_horizontal=1)

dir_ver = r"/mlcv1/WorkingSpace/Personal/baotg/Kalapa/argument_data/ver"
dir_hor = r"/mlcv1/WorkingSpace/Personal/baotg/Kalapa/argument_data/hor"

if not os.path.exists(dir_ver):
    os.mkdir(dir_ver)

if not os.path.exists(dir_hor):
    os.mkdir(dir_hor)


for folder_name in tqdm(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(("png", "jpg", "jpeg")):
            continue
        file_path = os.path.join(folder_path, file_name)
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

        path_ver = os.path.join(dir_ver, folder_name)
        path_hor = os.path.join(dir_hor, folder_name)

        if not os.path.exists(path_ver):
            os.mkdir(path_ver)

        if not os.path.exists(path_hor):
            os.mkdir(path_hor)
        print(file_path)

        augmented_img_ver = mytaco_ver.apply_vertical_taco(img, corruption_type="black")
        augmented_img_hor = mytaco_hor.apply_horizontal_taco(
            img, corruption_type="black"
        )

        file_ver = os.path.join(path_ver, file_name)
        file_hor = os.path.join(path_hor, file_name)
        cv.imwrite(file_hor, augmented_img_hor)
        cv.imwrite(file_ver, augmented_img_ver)
