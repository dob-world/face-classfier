import os
from glob import glob
from random import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
import PIL.Image as im
import PIL
import cv2

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


image_size = 64

def load_image(path):
    image = im.open(path)
    return image


class ImageDataset(Dataset):
    def __init__(self, celeba_dir, transform, noise_transform, length=999999, is_predict=False):
        self.celeba_dir = celeba_dir
        path = os.path.join(celeba_dir, '*.jpg')
        self.file_list = glob(path)
        self.transform = transform
        self.noise_transform = noise_transform
        self.length =  min(length, len(self.file_list))
        self.is_predict = is_predict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = cv2.imread(self.file_list[idx], cv2.COLOR_BGR2RGB)
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
    dataset = ImageDataset(celeba_dir,
                       transform=transf,
                       noise_transform=noise_transf)

    data_loader = DataLoader(dataset, batch_size=25)

    
    rows = 5
    cols = 5

    width=5
    height=5
    axes = []
    fig=plt.figure()

    for i, d in enumerate(data_loader):
        x_, y = d
        for i, x in enumerate(x_):
            x = to_pil_image(x)
            b, g, r = x.split()
            x = im.merge("RGB", (r, g, b))
            axes.append( fig.add_subplot(rows, cols, i+1) )
            axes[i].axis('off')
            plt.imshow(x)
            if i > 25:
                break
        break

    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()