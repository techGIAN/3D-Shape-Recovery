import torch
from torch.cuda import amp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from depth_modules import RelDepthModel
from dp_loss import DepthPredictionLoss
import cv2
import os
import numpy
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join


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
        sample = {'image': image}
        return sample


def train_depth():
    # can change the backbone to resnext101
    model = RelDepthModel(backbone='resnet50').to('cuda')

    img_dir = '../../rgb/'
    img_dir_out = 'rgb_depth_out/'
    img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    img_csv = pd.DataFrame({'ID': img_files})
    os.makedirs(img_dir_out, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_rgb_dataset = CustomDepthDataset(img_csv, img_dir, transform)

    # for ground truths
    img_gt_dir = '../../depth_zbuffer/'
    img_gt_dir_out = 'gt_depth_out/'
    img_gt_files = [f for f in listdir(img_gt_dir) if isfile(join(img_gt_dir, f))]
    img_gt_csv = pd.DataFrame({'ID': img_gt_files})
    os.makedirs(img_gt_dir_out, exist_ok=True)

    img_gt_depth_rgb_dataset = CustomDepthDataset(img_gt_csv, img_gt_dir, transform)
    data_gt_set = DataLoader(img_gt_depth_rgb_dataset, shuffle=True)

    for k, (d_p, d_g) in enumerate(zip(img_rgb_dataset, img_gt_depth_rgb_dataset)):

        print('Processing Image ' + str(k))
        rgb_image = d_p['image']
        d1 = rgb_image.shape[0]
        d2 = rgb_image.shape[1]
        rgb_copy = rgb_image[:, :, ::-1].copy()
        resized_copy = cv2.resize(rgb_copy, (448, 448))  # resize images to 448 x 448
        rgb_flip_50 = cv2.resize(rgb_image, (int(d2 // 2), int(d1 // 2)))  # flip images horizontally 50% of the time

        new_img = rgb_flip_50
        if len(new_img.shape) == 2:
            new_img = new_img[numpy.newaxis, :, :]  # add a new dimension
        if new_img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor()])
            new_img = transform(new_img)
        else:
            new_img = new_img.astype(numpy.float32)
            new_img = torch.from_numpy(new_img)
        scaled_img = new_img[None, :, :, :]

        rgb_gt_image = d_g['image']
        d1 = rgb_gt_image.shape[0]
        d2 = rgb_gt_image.shape[1]
        rgb_gt_copy = rgb_gt_image[:, :, ::-1].copy()
        resized_gt_copy = cv2.resize(rgb_gt_copy, (448, 448))  # resize images to 448 x 448
        rgb_gt_flip_50 = cv2.resize(rgb_gt_copy,
                                    (int(d2 // 2), int(d1 // 2)))  # flip images horizontally 50% of the time

        scaled_gt_img = rgb_gt_flip_50
        with amp.autocast(enabled=True):
            pred_depth = model.predict_depth(scaled_img).cpu().numpy().squeeze()

            gray_transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor()])
            gt_depth = gray_transform(scaled_gt_img).cpu().numpy().squeeze()

            dpl = DepthPredictionLoss(pr_d=pred_depth, gt_d=gt_depth)
            depth_loss = dpl.overall_loss  # either one of this
            # depth_loss = dpl.overall_loss().cuda()
            print('Image has loss ' + str(numpy.round(depth_loss.item(), 4)))

        optimizer.zero_grad()
        loss = Variable(depth_loss.float(), requires_grad=True).cuda()
        loss.backward()
        optimizer.step()

        if k % 1000 == 0:
            scheduler.step()
    torch.save(model, "saved_dpm")
