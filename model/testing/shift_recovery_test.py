import torchvision.transforms as transforms
import numpy
import cv2
import torch
from model.pcm_util import data_prepare
from model.depth_modules import RelDepthModel

dir = '../../DIODE/indoors/scene_00019/scan_00183'

# can change the backbone to resnext101


def test_shift_recovert():
    model = RelDepthModel(backbone='resnet50').to('cuda')
    model.load_state_dict(torch.load("../../saved_model/saved_dpm").state_dict())
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # rgb_image = cv2.imread(img)
    z = []
    z_base = []
    z_shift = []

    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and not 'depth' in filename and not '.mpy' in filename:
            ground_truth = numpy.load(f[:-4] + '_depth.npy')
            path = f
            rgb_image = cv2.imread(path)
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

            input = data_prepare(pred_depth, 2, 0)
            shift = model(input)

            our_depth = pred_depth - shift.item()
            z.append(numpy.average(ground_truth))
            z_base.append(numpy.average(pred_depth))
            z_shift.append(numpy.average(our_depth))

    import os
    dir = '../../DIODE/indoors/scene_00019/scan_00183'
    i = 0
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and not 'depth' in filename and not '.mpy' in filename:
            print(f"{i}: {f}")
            print(f"{i}: {f[:-4] + '_depth.npy'}")
            i += 1

