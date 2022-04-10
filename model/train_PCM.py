import pandas as pd
from torch.cuda import amp
import torch
import random
from SPVCNN import SPVCNN
from data_loader import DepthDataset
from pcm_util import data_prepare


def train_focal():
    model = SPVCNN(input_channel=5,
                   num_classes=1,
                   cr=1.0,
                   pres=0.01,
                   vres=0.01
                   ).to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=True)

    csv_file = pd.read_csv('dataset.csv')
    img_folder = '../depth_zbuffer/'


    train_dataset = DepthDataset(csv_file, img_folder)

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

if __name__ == '__main__':
    pass