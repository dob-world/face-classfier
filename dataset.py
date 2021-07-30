import os
from glob import glob
from random import random
import torch

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
import PIL.Image as im
import cv2
image_size = 64

def load_image(path):
    image = im.open(path)
    return image


class ImageDataset(Dataset):
    def __init__(self, celeba_dir, transform, noise_transform, length=120000, is_predict=False):
        self.celeba_dir = celeba_dir
        path = os.path.join(celeba_dir, '*.png')
        self.file_list = glob(path)
        self.transform = transform
        self.noise_transform = noise_transform
        self.length =  min(length, len(self.file_list))
        self.is_predict = is_predict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = cv2.imread(self.file_list[idx], cv2.IMREAD_COLOR)

        if self.is_predict:
            image = self.transform(image)
            return image, self.file_list[idx]

        if random() > 0.5:
            label = 1.
            if not self.transform == None:
                image = self.transform(image)
        else:
            label = 0.
            if not self.noise_transform == None:
                image = self.noise_transform(image)

        
        return image, np.float32(label)


def transf(image):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((1, 1, 1), (-1, -1, -1)),
    ])(image)


def noise_transf(image):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop((image_size, image_size)),
        transforms.Normalize((1, 1, 1), (-1, -1, -1)),
    ])(image)


if __name__ == "__main__":
    celeba_dir = '../../dataset/celeba/img_align_celeba'
    others_dir = '../../dataset/others'
    image_size = 64
    dataset = ImageDataset(celeba_dir,
                           transform=transf,
                           noise_transform=noise_transf,
                           )
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(torch.min(train_features))
    print(torch.max(train_features))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().permute(1, 2, 0)
    img = img.numpy()
    label = train_labels[0]
    cv2.imshow('test', img)
    cv2.waitKey()
    print(f"Label: {label}")
