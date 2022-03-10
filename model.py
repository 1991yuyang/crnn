import torch as t
from torch import nn
from torchvision import models
# how to use nn.CTCLoss: https://zhuanlan.zhihu.com/p/67415439


class CRNN(nn.Module):

    def __init__(self, num_classes, input_h):
        """

        :param num_classes: count of charactor, include blank
        :param input_h: height of input image
        """
        super(CRNN, self).__init__()
        assert input_h % 16 == 0, "input_h should be a multiple of 16"
        self.input_h = input_h
        self.num_classes = num_classes
        model = models.vgg11(pretrained=True).features
        model.__setattr__("10", nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.__setattr__("15", nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.__setattr__("16", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0))
        feature_block = list(model.children())
        feature_block.insert(12, nn.BatchNorm2d(num_features=512))
        feature_block.insert(15, nn.BatchNorm2d(num_features=512))
        self.cnn_features = nn.Sequential(*feature_block[:-4])
        self.bilstm1 = nn.GRU(int(input_h / 16 - 1) * 512, 256, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=512, out_features=512)
        self.bilstm2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.clsf = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        assert x.size()[2] == self.input_h, "height of input x should be input_h"
        cnn_feature = self.cnn_features(x).view((x.size()[0], 512 * int(self.input_h / 16 - 1), -1)).permute(dims=[0, 2, 1]).contiguous()  # N, W, 512 * (self.input_h / 16 - 1)
        lstm_feature1, _ = self.bilstm1(cnn_feature)  # N, W, 512
        embedding1 = self.linear(lstm_feature1.contiguous().view((-1, 512))).contiguous().view((x.size()[0], -1, 512))  # N, W, 512
        lstm_feature2, _ = self.bilstm2(embedding1)  # N, W, 512
        output = self.clsf(lstm_feature2.contiguous().view(-1, 512)).contiguous().view((x.size()[0], -1, self.num_classes))  # N, W, num_classes
        return output


if __name__ == "__main__":
    d = t.randn(4, 3, 32, 100)
    model = CRNN(num_classes=10, input_h=32)
    for i in range(10):
        output = model(d)
        print(output.size())
