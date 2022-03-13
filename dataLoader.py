from torch.utils import data
import os
from PIL import Image, ImageOps
from generate_data import generate_one_image
import json
import numpy as np
from torchvision import transforms as T
import torch as t
from numpy import random as rd


"""
charactors.txt: all charactors in one line split by ',', for example: 1,2,3,4,5
img_dir:
    1.jpg
    2.jpg
    ......
    
label_pth: xxx/xxx/xxx.json
xxx.json format:
{
    "1.jpg": text1,
    "2.jpg": text2,
    ......
}
"""


class MySet(data.Dataset):

    def __init__(self, input_h, img_dir, label_pth, is_train):
        self.img_dir = img_dir
        self.input_h = input_h
        self.labels = json.load(open(label_pth))
        with open("charactors.txt", "r", encoding="utf-8") as file:
            self.all_charactors = file.read().split(",")
        self.img_names = os.listdir(img_dir)
        if is_train:
            self.transformer = T.Compose([
                T.RandomRotation(8),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                KeepRatioResize(input_h=self.input_h)
            ])
        else:
            self.transformer = KeepRatioResize(input_h=self.input_h)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth)
        img = self.transformer(img)
        w, h = img.size
        label = [self.all_charactors.index(c) for c in list(self.labels[img_name])]
        return img, label, w

    def __len__(self):
        return len(self.img_names)


# class KeepRatioResize(object):
# 
#     def __init__(self, input_h, interpolation=Image.BILINEAR):
#         self.input_h = input_h
#         self.interpolation = interpolation
# 
#     def __call__(self, img):
#         original_w, original_h = img.size
#         w_h_ratio = original_w / original_h
#         input_w = int(w_h_ratio * self.input_h)
#         img = img.resize((input_w, self.input_h), self.interpolation)
#         return img


class KeepRatioResize(object):

    def __init__(self, input_h, interpolation=Image.BILINEAR):
        self.input_h = input_h
        self.interpolation = interpolation
        self.pad = PadOp()

    def __call__(self, img):
        original_w, original_h = img.size
        w_h_ratio = original_w / original_h
        input_w = int(w_h_ratio * self.input_h)
        if input_w == 0:
            input_w = 1
        img = img.resize((input_w, self.input_h), self.interpolation)
        if input_w < 8:
            pad_size = 8 - input_w
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            img = self.pad(img, (left_pad, 0, right_pad, 0))
        return img



class PadOp(object):

    def __init__(self):
        pass

    def __call__(self, img, padding):
        """

        :param img:
        :param padding: left, top, right and bottom
        :return:
        """
        img = ImageOps.expand(img, border=padding)
        return img


def return_collate_fn():
    to_tensor = T.ToTensor()
    pad_op = PadOp()
    def my_collate_fn(batch):
        images = []
        labels = []
        labels_seperate = []
        lenghts = []
        max_w = np.max([item[-1] for item in batch])
        for img, label, w in batch:
            w, h = img.size
            paddsize = max_w - w
            if paddsize >= 2:
                left_padd = paddsize // rd.randint(2, paddsize + 1)
                right_padd = paddsize - left_padd
            else:
                left_padd = rd.choice([paddsize, 0])
                right_padd = paddsize - left_padd
            img = pad_op(img, (left_padd, 0, right_padd, 0))
            img = to_tensor(img)
            images.append(img.unsqueeze(0))
            labels_seperate.append(label)
            labels.extend(label)
            lenghts.append(len(label))
        images = t.cat(images, dim=0)
        return images, t.tensor(labels).type(t.LongTensor), tuple(lenghts), labels_seperate
    return my_collate_fn


def make_loader(input_h, img_dir, label_pth, is_train, batch_size, num_workers):
    loader = iter(data.DataLoader(MySet(input_h, img_dir, label_pth, is_train), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=return_collate_fn(), num_workers=num_workers))
    return loader


if __name__ == "__main__":
    s = MySet(input_h=32, img_dir=r"/home/yuyang/data/crnn_data/train_image", label_pth=r"/home/yuyang/data/crnn_data/train_label.json", is_train=True)
    loader = make_loader(input_h=32, img_dir=r"/home/yuyang/data/crnn_data/train_image", label_pth=r"/home/yuyang/data/crnn_data/train_label.json", is_train=True, batch_size=4)
    for img, label, lenghts in loader:
        print(img.size())
        input()
