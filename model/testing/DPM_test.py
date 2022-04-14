import torchvision.transforms as transforms
import numpy
import cv2
import torch
from model.depth_modules import RelDepthModel
from model.network_auxiliary_path import resnet50_stride32, resnext101_stride32x8d
import os


def DPM_test(backbone=50):
    # can change the backbone to resnext101
    model = RelDepthModel(backbone='resnet50').to('cuda')
    model.load_state_dict(torch.load("../../saved_model/saved_dpm").state_dict())
    if backbone == 50:
        model.encoder = resnet50_stride32()
    else:
        model.encoder = resnext101_stride32x8d
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    i = 0
    # rgb_image = cv2.imread(img)
    z = []
    z_hat = []

    depth_dir = os.listdir('../../depth')
    rgb_dir = os.listdir('../../rgb')

    for rgb, depth in zip(rgb_dir, depth_dir):
        f1 = os.path.join('../../rgb', rgb)
        f2 = os.path.join('../../depth', depth)
        # checking if it is a file
        if os.path.isfile(f):
            ground_truth = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)

            rgb_image = cv2.imread(f1)
            d1 = rgb_image.shape[0]
            d2 = rgb_image.shape[1]
            rgb_copy = rgb_image[:, :, ::-1].copy()
            resized_copy = cv2.resize(rgb_copy, (448, 448))  # resize images to 448 x 448
            rgb_flip_50 = cv2.resize(rgb_image,
                                     (int(d2 // 2), int(d1 // 2)))  # flip images horizontally 50% of the time
            ground_truth = cv2.resize(rgb_image, (int(d2 // 2), int(d1 // 2)))
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
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
            pred_depth = model.predict_depth(scaled_img).cpu().numpy().squeeze()

            z.append(numpy.average(ground_truth))
            z_hat.append(numpy.average(pred_depth))
            i += 1
            print(i)