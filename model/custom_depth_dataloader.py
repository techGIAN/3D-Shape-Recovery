import csv

import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader, Dataset
import os
import cv2


class CustomDepthDataset(Dataset):
    def __init__(self, csv, img_folder, transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.img_names = self.csv[:]['ID']

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image = cv2.imread(self.img_folder + self.img_names.iloc[index])
        image = self.transform(image)
        sample = {'image': image}
        return sample