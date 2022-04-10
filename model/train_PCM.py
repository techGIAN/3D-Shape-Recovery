import pandas as pd
from torch.cuda import amp
import torch
import random
from SPVCNN import SPVCNN
from data_loader import DepthDataset
from pcm_util import data_prepare
from torch.optim.lr_scheduler import StepLR

def train_focal():
    model = SPVCNN(input_channel=5,
                   num_classes=1,
                   cr=1.0,
                   pres=0.01,
                   vres=0.01
                   ).to("cuda")

    learning_rate = 0.24
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    scaler = amp.GradScaler(enabled=True)

    csv_file = pd.read_csv('dataset.csv')
    img_folder = '../../depth_zbuffer/'


    train_dataset = DepthDataset(csv_file, img_folder)
    for k, feed_dict in enumerate(train_dataset):
        print(optimizer.param_groups[0]['lr'])
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

        if k % 1000 == 0:
          scheduler.step()
    torch.save(model, "saved_pcm_focal")

def train_shift():
    model = SPVCNN(input_channel=3,
                   num_classes=1,
                   cr=1.0,
                   pres=0.01,
                   vres=0.01
                   ).to("cuda")

    learning_rate = 0.24
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    scaler = amp.GradScaler(enabled=True)

    csv_file = pd.read_csv('dataset.csv')
    img_folder = '../../depth_zbuffer/'


    train_dataset = DepthDataset(csv_file, img_folder)
    for k, feed_dict in enumerate(train_dataset):
        print(optimizer.param_groups[0]['lr'])
        delta = round(random.uniform(-0.25, 0.8),2)
        img = feed_dict['image']
        input = data_prepare(img, img, 2, delta)
        with amp.autocast(enabled=True):
          outputs = model(input)
          loss = abs(outputs - delta)
        print(f'[step {k + 1}] loss = {loss.item()}')
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if k % 1000 == 0:
          scheduler.step()
    torch.save(model, "saved_pcm_shift")

if __name__ == '__main__':
    pass