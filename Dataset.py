import torch
import torch.utils.data as data
import os
import cv2
import torchvision.transforms.transforms as transforms

class SRCNN_Dataset(data.Dataset): 
    def __init__(self, transform=None, train=True):
        parent_path = './dataset/train'

        if not train:
            parent_path ='./dataset/val'

        hr_path = os.path.join(parent_path, "hr")
        lr_path = os.path.join(parent_path, "lr")

        hr_list = os.listdir(hr_path)
        lr_list = os.listdir(lr_path)

        self.hr_imgs = list()
        self.lr_imgs = list()

        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        for hr_img in hr_list:
            img = cv2.imread(os.path.join(hr_path, hr_img))

            if transform is not None:
                img = self.transform(img)

            img = self.to_tensor(img)

            self.hr_imgs.append(img)

        for lr_img in lr_list:
            img = cv2.imread(os.path.join(lr_path, lr_img))

            if transform is not None:
                img = self.transform(img)

            img = self.to_tensor(img)

            self.lr_imgs.append(img)

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, idx):
        return self.hr_imgs[idx], self.lr_imgs[idx]