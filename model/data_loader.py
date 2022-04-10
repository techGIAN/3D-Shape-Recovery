import csv

import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader, Dataset
import os
import cv2



def filenames_in_csv(FOLDER_NAME, FILE_NAME):
    path = FOLDER_NAME
    with open(FILE_NAME, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID'])
        for root, dirs, files in os.walk(path):
            for filename in files:
                writer.writerow([filename])


class DepthDataset(Dataset):
    def __init__(self, csv, img_folder):
        self.csv = csv
        # self.transform = transform
        self.img_folder = img_folder
        self.img_names = self.csv[:]['ID']

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image = cv2.imread(self.img_folder + self.img_names.iloc[index], cv2.IMREAD_GRAYSCALE)
        # image = self.transform(image)
        sample = {'image': image}

        return sample

