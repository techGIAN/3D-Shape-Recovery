import csv

import torch
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader,Dataset
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filenames_in_csv(FOLDER_NAME, FILE_NAME):
    path = FOLDER_NAME
    with open(FILE_NAME, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID'])
        for root, dirs, files in os.walk(path):
            for filename in files:
                writer.writerow([filename])

class DepthDataset(Dataset):
    def __init__(self, csv, img_folder, transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.img_names = self.csv[:]['ID']

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image = cv2.imread(self.img_folder + self.img_names.iloc[index], cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        sample = {'image': image}

        return sample


if __name__ == '__main__':
    csv_file  = pd.read_csv('dataset.csv')
    BATCH_SIZE = 40
    img_folder = '../dataset/depth_zbuffer/'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = DepthDataset(csv_file, img_folder, transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle=True
    )


    def imshow(inp, title=None):
        """imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.savefig('1.png')

    images = next(iter(train_dataloader))
    output = torchvision.utils.make_grid(images['image'])
