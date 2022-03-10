from numpy import random as rd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
from tqdm import tqdm


img_save_dir = r"/home/yuyang/data/crnn_data/train_image"  # generated image save dir
labels_pth = r"/home/yuyang/data/crnn_data/train_label.json"  # generated label save dir, .json file
img_height_min = 30  # minimum height of image
img_height_max = 50  # maxmum height of image
img_width_min = 95  # minimum width of image
img_width_max = 200  # maxmum width of image
blank_index = 0  # blank index in charactor.txt
charactor_count_min = 4  # minimum count of charactor in one image
charactor_count_max = 8 # maxmum count of charactor in one image
sample_count = 8000  # how many sample
fonts = [
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
         ]  # change your font file
with open("charactors.txt", "r", encoding="utf-8") as file:
    charactors = file.read().split(",")
charactors.pop(blank_index)
labels = {}


def generate_one_image():
    """

    :return: image: PIL image
    labels: text string of generated image like "abcde.."
    """
    charactor_count = rd.randint(charactor_count_min, charactor_count_max + 1)
    img_height = rd.randint(img_height_min, img_height_max + 1)
    img_width = rd.randint(img_width_min, img_width_max + 1)
    image = rd.randint(130, 180, (img_height, img_width, 3), dtype=np.uint8)
    image = Image.fromarray(image)
    image_draw = ImageDraw.Draw(image)
    with_of_every_charactor = img_width // charactor_count
    label = ""
    for i in range(charactor_count):
        color_black = rd.randint(0, 30)
        color_white = rd.randint(185, 255)
        color = rd.choice([color_white, color_black])
        charactor = rd.choice(charactors)
        label += charactor
        charactor_w_start = rd.randint(i * with_of_every_charactor, i * with_of_every_charactor + with_of_every_charactor // 10 + 1)
        charactor_h_start = rd.randint(img_height // 10, img_height // 7)
        font_size = min(img_height - charactor_h_start, (i + 1) * with_of_every_charactor - charactor_w_start)
        font = ImageFont.truetype(rd.choice(fonts), font_size, encoding="utf-8")
        image_draw.text((charactor_w_start, charactor_h_start), charactor, (color,) * 3, font=font)
    return image, label


def main():
    for i in tqdm(range(sample_count)):
        image_name = "%d.jpg" % (i,)
        image_save_path = os.path.join(img_save_dir, image_name)
        try:
            image, label = generate_one_image()
        except:
            continue
        else:
            labels[image_name] = label
            image.save(image_save_path)
            time.sleep(0.1)
    json.dump(labels, open(labels_pth, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()