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
    model = SPVCNN(input_channel=5,
                   num_classes=1,
                   cr=1.0,
                   pres=0.01,
                   vres=0.01
                   )
    model.eval()
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=True)

    csv_file = pd.read_csv('dataset.csv')
    BATCH_SIZE = 40
    img_folder = '../depth_zbuffer/'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = DepthDataset(csv_file, img_folder, None)

    dataflow = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    for k, feed_dict in enumerate(train_dataset):
        alpha = round(random.uniform(0.6, 1.25), 2)
        img = feed_dict['image']
        input = data_prepare(img, img, 1, alpha)
        with amp.autocast(enabled=True):
            outputs = model(input)
            loss = abs(outputs - alpha)

        print(f'[step {k + 1}] loss = {loss.item()}')

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
