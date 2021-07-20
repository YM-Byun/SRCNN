import torch
import torchvision.transforms as transforms
import time
import sys
import torch.nn as nn
import argparse

from Dataset import SRCNN_Dataset
from torch.utils.data import DataLoader
from model import SRCNN

batch_size=16
momentum=0.9
weight_decay = 0.005
learning_rate = 0.0001
epochs = 300
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser(description='VGG16')
    parser.add_argument('--gpu', type=int, default=-1,
            help='gpu number')

    args = parser.parse_args()

    return args

def get_mean_std(dataset):
    mean = dataset.data.mean(axis=(0,1,2,)) / 255
    std = dataset.data.std(axis=(0,1,2,)) / 255

    return mean, std

def main():
    global device

    parser = get_parser()

    if (parser.gpu != -1):
        device = torch.device('cuda:' + str(parser.gpu))

    print ("\nLoading Dataset...")

    dataset = SRCNN_Dataset(train=True)

    mean, std = get_mean_std(dataset)

    print (f"Mean: {mean}  | Std: {std}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    train_dataset = SRCNN_Dataset(train=True, transform=transform)
    val_dataset =SRCNN_Dataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4)

    print ("\nLoaded Dataset!")

    print ("\n========================================\n")

    srcnn = SRCNN()

    global learning_rate

    optimizer = torch.optim.SGD(srcnn.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250])

    best_loss = 9.0

    if is_cuda:
        srcnn.to(device)
        criterion = criterion.to(device)

    for epoch in range(epochs):

        train(train_loader, srcnn, criterion, optimizer, scheduler, epoch)

        print ("")

        loss = validate(val_loader, srcnn, criterion, epoch)

        is_best = False

        if loss < best_loss:
            is_best = True
            best_loss = loss

        if is_best:
            torch.save(srcnn.state_dict(), "./weight/best_weight.pth")
            print (f"\nSave best model at loss: {loss:.4f}!")

        print ("\n========================================\n")

        torch.save(srcnn.state_dict(), "./weight/lastest_weight.pth")

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

        if (i % 50 == 49) or (i == len(train_loader) - 1):
            print (f"Epoch [{epoch+1}/{epochs}] | Train iter [{i+1}/{len(train_loader)}] | loss = {(running_loss / float(i+1)):.5f} | lr = {get_lr(optimizer):.5f}")

    scheduler.step()

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, label = data

            if is_cuda:
                inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()


    print (f"Epoch [{epoch+1}/{epochs}] | Validation |bloss = {(running_loss / float(i)):.5f}")

    return (running_loss / float(i))

def get_lr(optimizer):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break

    return lr

if __name__ == '__main__':
    main()
