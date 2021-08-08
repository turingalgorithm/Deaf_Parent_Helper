import torch
import torch.nn as nn
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter
from utils.AverageMeter import AverageMeter
from sklearn.metrics import confusion_matrix, classification_report
from functools import reduce

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res,pred
def train(model: torch.nn.Module, train_loader: Iterable,
          optimizer: torch.optim.Optimizer, epoch: int, summary: SummaryWriter):
    model.train()
    sc = torch.tensor([0.2, 0.2, 0.4, 0.4]).cuda()
    loss = nn.CrossEntropyLoss(weight=sc)
    train_loss = AverageMeter()
    for step, data in enumerate(train_loader):
        audio, label = data
        audio = audio.cuda()
        label = label.squeeze()
        label = label.cuda()
        pred = model(audio)
        losses = loss(pred, label)
        train_loss.update(losses.item(), audio.size()[0])
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print("train losses : {} , epoch : {}".format(losses, epoch))
    summary.add_scalar('train/loss', train_loss.avg, epoch)

def val(model: torch.nn.Module, val_loader: Iterable, epoch: int, summary: SummaryWriter):
    model.eval()
    val_acc = AverageMeter()
    val_losses = AverageMeter()
    temp = []
    pred_list = []

    with torch.no_grad():
        loss = nn.CrossEntropyLoss()
        for step, data in enumerate(val_loader):
            audio, label = data
            audio = audio.cuda()
            label = label.cuda()

            label = label.squeeze()

            #label = label.unsqueeze(0)
            pred = model(audio)
            losses = loss(pred, label)

            prec1, preds = accuracy(pred.data, label)
            temp.append(label.tolist())
            pred_list.append(preds.tolist())
            val_losses.update(losses.item(), audio.size(0))
            val_acc.update(prec1[0], audio.size(0))
        print("losses : {} , acc  : {} , epoch : {}".format(val_losses.avg, val_acc.avg, epoch))

    # confusion_matrix

    y_true = reduce(lambda x, y: x + y, temp)


    y_pred = []

    for i in pred_list:
        temp = i[0]
        for k in temp:
            y_pred.append(k)


    confusion_matrixs = confusion_matrix(y_true, y_pred)
    print(confusion_matrixs)
    print(classification_report(y_true, y_pred))

    summary.add_scalar('val/loss', val_losses.avg, epoch)
    summary.add_scalar('val/acc', val_acc.avg, epoch)
