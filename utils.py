import torch as t
from torch import nn


softmax_op = nn.Softmax(dim=2)


def decode_one_predicton_result(model_output, blank_index):
    """

    :param blank_index: blank index of charactors
    :param model_output: one image model output, shape: [T, 1, num_classes]
    :return: charactor index list of current image
    """
    softmax_result = softmax_op(model_output)
    indexs = t.argmax(softmax_result.squeeze(1), dim=1).detach().cpu().numpy().tolist()
    indexs_length = len(indexs)
    result = []
    for i in range(indexs_length):
        if i < indexs_length - 1:
            if indexs[i] != indexs[i + 1] and indexs[i] != blank_index:
                result.append(indexs[i])
        else:
            if indexs[i] != blank_index:
                result.append(indexs[i])
    return result

