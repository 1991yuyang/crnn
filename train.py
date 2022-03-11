from model import CRNN
from dataLoader import make_loader
import torch as t
from torch import nn, optim
from torch.nn import functional as F
import os
from visdom import Visdom
from utils import decode_one_predicton_result
import numpy as np
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
epoch = 3000
batch_size = 48
init_lr = 0.005
min_lr = 0.0001
cosine_lr_sch_cycle_times = 0.5
input_h = 32  # input height of image
train_img_dir = r"/home/yuyang/data/crnn_data/train_image"
valid_img_dir = r"/home/yuyang/data/crnn_data/valid_image"
train_label_pth = r"/home/yuyang/data/crnn_data/train_label.json"
valid_label_pth = r"/home/yuyang/data/crnn_data/valid_label.json"
print_step = 10
blank_index = 0  # blank charactor index in charactors
num_workers = 4
weight_decay = 0.0001
with open("charactors.txt", "r", encoding="utf-8") as file:
    charactors = file.read().split(",")
num_classes = len(charactors)
best_valid_loss = float("inf")
train_loss_window = Visdom()
valid_loss_window = Visdom()
total_step = 1


def calc_accu(model_output, targets):
    batch_predict_charactor_indexs = []
    for i in range(model_output.size()[1]):
        one_output = model_output[:, i:i + 1, :]
        result = decode_one_predicton_result(one_output, blank_index)
        batch_predict_charactor_indexs.extend(result)
    if len(batch_predict_charactor_indexs) != len(targets):
        return float(0)
    accu = np.sum((np.array(batch_predict_charactor_indexs) == targets.detach().cpu().numpy())) / len(batch_predict_charactor_indexs)
    return float(accu)


def train_epoch(current_epoch, model, criterion, optimizer, train_loader):
    global total_step
    model.train()
    step = len(train_loader)
    current_step = 1
    for d_train, targets, target_lenghts in train_loader:
        d_train_cuda = d_train.cuda(device_ids[0])
        targets = targets.cuda(device_ids[0])
        train_output = model(d_train_cuda)  # N, W, num_classes
        train_output = train_output.permute(dims=[1, 0, 2]) # W, N, num_classes
        T = train_output.size()[0]
        log_probs = F.log_softmax(train_output, dim=2).requires_grad_()
        input_lenghts = (T,) * batch_size
        train_loss = criterion(log_probs, targets, input_lenghts, target_lenghts)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if current_step % print_step == 0:
            train_accu = calc_accu(train_output, targets)
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, train_accu:%.5f" % (current_epoch, epoch, current_step, step, train_loss.item(), train_accu))
            train_loss_window.line([train_loss.item()], [total_step], win="train loss", update="append", opts=dict(title="train_loss"))
        current_step += 1
        total_step += 1
    print("saving epoch model")
    t.save(model.state_dict(), "epoch.pth")
    return model


def valid_epoch(current_epoch, model, criterion, valid_loader):
    global best_valid_loss
    model.eval()
    step = len(valid_loader)
    accum_loss = 0
    accum_accu = 0
    for d_valid, targets, target_lenghts in valid_loader:
        d_valid_cuda = d_valid.cuda(device_ids[0])
        targets = targets.cuda(device_ids[0])
        with t.no_grad():
            valid_output = model(d_valid_cuda)  # N, W, num_classes
            valid_output = valid_output.permute(dims=[1, 0, 2]) # W, N, num_classes
            T = valid_output.size()[0]
            log_probs = F.log_softmax(valid_output, dim=2).requires_grad_()
            input_lenghts = (T,) * batch_size
            valid_loss = criterion(log_probs, targets, input_lenghts, target_lenghts)
            accum_loss += valid_loss.item()
            valid_accu = calc_accu(valid_output, targets)
            accum_accu += valid_accu
    avg_loss = accum_loss / step
    avg_accu = accum_accu / step
    valid_loss_window.line([avg_loss], [current_epoch], win="valid loss", update="append", opts=dict(title="valid_loss"))
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        print("saving best model......")
        t.save(model.state_dict(), "best.pth")
    print("##########valid epoch:%d############" % (current_epoch,))
    print("epoch:%d/%d, valid_loss:%.5f, valid_accu:%.5f" % (current_epoch, epoch, avg_loss, avg_accu))
    return model


def main():
    model = CRNN(num_classes=num_classes, input_h=input_h)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    criterion = nn.CTCLoss(blank=blank_index).cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch // int(2 * cosine_lr_sch_cycle_times), eta_min=min_lr)
    for e in range(epoch):
        print("lr:%.5f" % (lr_sch.get_lr()[0],))
        current_epoch = e + 1
        train_loader = make_loader(input_h, train_img_dir, train_label_pth, True, batch_size, num_workers)
        valid_loader = make_loader(input_h, valid_img_dir, valid_label_pth, False, batch_size, num_workers)
        model = train_epoch(current_epoch, model, criterion, optimizer, train_loader)
        model = valid_epoch(current_epoch, model, criterion, valid_loader)
        lr_sch.step()


if __name__ == "__main__":
    main()
