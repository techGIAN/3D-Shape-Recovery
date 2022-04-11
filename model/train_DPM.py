import torch
from torch.cuda import amp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import random
from depth_modules import RelDepthModel
from dp_loss import DepthPredictionLoss
from custom_depth_dataloader import CustomDepthDataset

import os
from os import listdir
from os.path import isfile, join

def train_depth():
    # can change the backbone to resnext101
    model = RelDepthModel(backbone='resnet50')
    model.eval()
    model.cuda()

    img_dir = '../dataset/rgb/'
    img_dir_out = '../rgb_depth_out/'
    img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    img_csv = pd.DataFrame({'image': img_files})
    os.makedirs(img_dir_out, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=0.1)
    scaler = amp.GradScaler(enabled=True)

    img_rgb_dataset = CustomDepthDataset(img_csv, img_dir, None)
    data_set = DataLoader(img_rgb_dataset, batch_size=40, shuffle=True)



    for k, d in enumerate(data_set):
        print('Processing Image ' + str(k))
        
        rgb_image = cv2.imread(d)
        d1 = rgb_image.shape[0]
        d2 = rgb_image.shape[1]

        rgb_copy = rgb[:,:,::-1].copy()
        resized_copy = cv2.resize(rgb_copy, (448, 448)) # resize images to 448 x 448
        rgb_flip_50 = cv2.resize(rgb_image, (int(d2//2), int(d1//2))) # flip images horizontally 50% of the time

        new_img = rgb_flip_50
        if len(new_img) == 2:
            new_img = resized_copy[np.newaxis,:,:] # add a new dimension
        if new_img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor()])
            new_img = transform(new_img)
        else:
            new_img = new_img.astype(np.float32)
            new_img = torch.from_numpy(img)
        
        scaled_img = new_img[None,:,:,:]
        
        with amp.autocast(enabled=True):
            pred_depth = model.predict_depth(scaled_img).cpu().numpy().squeeze()
            gt_depth = None # where to find from data?

            dpl = DepthPredictionLoss(pred_depth, gt_depth)
            depth_loss = dpl.overall_loss().to('cuda') # either one of this
            # depth_loss = dpl.overall_loss().cuda()

        print('Image has loss ' + str(round(depth_loss, 4)))

        optimizer.zero_grad()
        scaler.scale(depth_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.save(model, "saved_dpm")
