import os
import random
from multiprocessing import Process

batch = 1
img_num = 1000
# input_file = "/mlcv/WorkingSpace/Personals/baotg/BKAI/TaskKiller/TextRecognitionDataGenerator/trdg/dicts/vi.txt"
output_dir = "/mlcv1/WorkingSpace/Personal/baotg/Kalapa/data/synthetic_tuongbck/data"
font_dir = "/mlcv/WorkingSpace/Personals/baotg/BKAI/TaskKiller/TextRecognitionDataGenerator/trdg/fonts/vi"


def generate_batch(_b, batch, img_num, output_dir, font_dir):
    background = random.choice([0, 1])
    f = random.randint(100, 128)
    length = random.randint(5, 8)
    k = random.randint(0, 4)
    sw = random.uniform(1, 1.3)
    width = random.randint(1800, 2000)
    color_range = random.choice(["#1d204a,#1e1d4a", "#1f1f1f,#282828"])

    os.system(
        "python /mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/gen-synthetic/TextRecognitionDataGenerator/trdg/run.py -c "
        + f'{img_num//batch} -tc "{color_range}" -al 0 -na 2 -l vi -sw {sw} -b {background} -f {f} --length {length} -k {k} -rk -d 3 -do 2 -t 16 --output_dir {output_dir}/{_b} '
        + f"-fd {font_dir} "
        + "-dt /mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/gen-synthetic/resources/vi.txt"    
    )


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# procs = []
# for _b in range(batch):
#     proc = Process(
#         target=generate_batch(_b, batch, img_num, input_file, output_dir, font_dir)
#     )
#     procs.append(proc)
#     proc.start()

for i in range(200):
    print("Running batch ", i)
    generate_batch(i, batch, img_num, output_dir, font_dir)