from model import CRNN
from dataLoader import KeepRatioResize
from PIL import Image
from torchvision import transforms as T
from torch import nn
import os
import torch as t
from utils import decode_one_predicton_result
import PIL
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_pth = r"/home/yuyang/data/crnn_data/valid_image/165.jpg"
blank_index = 0
input_h = 32
use_best_model = True
transformer = T.Compose([
    KeepRatioResize(input_h=input_h),
    T.ToTensor()
])
with open("charactors.txt", "r", encoding="utf-8") as file:
    charactors = file.read().split(",")
num_classes = len(charactors)


def load_one_image(img_pth):
    if isinstance(img_pth, str):
        image_pil = Image.open(img_pth)
    elif isinstance(img_pth, PIL.Image.Image):
        image_pil = img_pth
    else:
        raise(Exception("img_pth should be string or PIL image"))
    image_tensor = transformer(image_pil).unsqueeze(0).cuda(0)
    return image_tensor, image_pil


def load_model():
    model = CRNN(num_classes=num_classes, input_h=input_h)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


model = load_model()


def predict_one_image(img_pth):
    """

    :param img_pth: string or PIL image
    :return:
    """
    image_tensor, image_pil = load_one_image(img_pth)
    with t.no_grad():
        output = model(image_tensor)  # 1, T, num_classes
        output = output.permute(dims=[1, 0, 2])  # T, 1, num_classes
        output_charactor_indexs = decode_one_predicton_result(output, blank_index)
        result = "".join([charactors[i] for i in output_charactor_indexs])
    return result


if __name__ == "__main__":
    result = predict_one_image(img_pth)
    print(result)
