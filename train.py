import torch
import torchvision.transforms as transforms
import time
import sys
import torch.nn as nn
import argparse

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import VGG16

batch_size=256
momentum=0.9
weight_decay = 0.005
learning_rate = 0.0001
epochs = 130
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def get_mean_std(dataset):
    mean = dataset.data.mean(axis=(0,1,2,)) / 255
    std = dataset.data.std(axis=(0,1,2,)) / 255

    return mean, std

def get_parser():
    parser = argparse.ArgumentParser(description='VGG16')
    parser.add_argument('--gpu', type=int, default=-1,
            help='gpu number')

    args = parser.parse_args()

    return args

def main():
    global device

    parser = get_parser()

    if (parser.gpu != -1):
        device = torch.device('cuda:' + str(parser.gpu))

    cifar10_dataset = CIFAR10(root='./dataset', train=True,
            download=True)

    mean, std = get_mean_std(cifar10_dataset)

    transform_v2 = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    print ("\nLoading Cifar 10 Dataset...")

    train_dataset = CIFAR10(root='./dataset', train=True, download=True,
        transform=transform_v2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4)

    val_dataset = CIFAR10(root='./dataset', train=False,
            download = True, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')

    print ("Loaded Cifar 10!\n")

    print ("\n========================================\n")

    vgg16 = VGG16()

    global learning_rate

    optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[55, 95, 115])

    best_acc = 0.0
    best_loss = 9.0

    if is_cuda:
        vgg16.to(device)
        criterion = criterion.to(device)

    for epoch in range(epochs):

        train(train_loader, vgg16, criterion, optimizer, scheduler, epoch)

        print ("")

        acc, loss = validate(val_loader, vgg16, criterion, epoch)

        is_best = False

        if best_acc == acc:
            if loss < best_loss:
                is_best = True
                best_loss = loss

        if best_acc < acc:
            is_best = True
            best_acc = acc

        if is_best:
            torch.save(vgg16.state_dict(), "./weight/best_weight.pth")
            print (f"\nSave best model at acc: {acc:.4f},  loss: {loss:.4f}!")

        print ("\n========================================\n")

        torch.save(vgg16.state_dict(), "./weight/lastest_weight.pth")

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, label = data

        if is_cuda:
            inputs, label = inputs.to(device), label.to(device)

        optimizer.zero_grad()

        outputs =  model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc1, acc5 = accuracy(outputs, label, topk=(1,5))

        if (i % 50 == 49) or (i == len(train_loader) - 1):
            print (f"Epoch [{epoch+1}/{epochs}] | Train iter [{i+1}/{len(train_loader)}] | acc1 = {acc1[0]:.3f} | acc5 = {acc5[0]:.3f} | loss = {(running_loss / float(i+1)):.5f} | lr = {get_lr(optimizer):.5f}")

    scheduler.step()

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0

    total_acc1 = 0.0
    total_acc5 = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, label = data

            if is_cuda:
                inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            acc1, acc5 = accuracy(outputs, label, topk=(1,5))

            total_acc1 += acc1
            total_acc5 += acc5

        total_acc1 /= len(val_loader)
        total_acc5 /= len(val_loader)

    print (f"Epoch [{epoch+1}/{epochs}] | Validation | acc1 = {total_acc1[0]:.3f} | acc5 = {total_acc5[0]:.3f} | loss = {(running_loss / float(i)):.5f}")

    return total_acc1[0], (running_loss / float(i))

def get_lr(optimizer):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break

    return lr

def accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _,pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res


if __name__ == '__main__':
    main()
