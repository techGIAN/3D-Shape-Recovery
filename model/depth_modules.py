'''
   Some functions borrowed ideas from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
'''
import network_auxiliary_path as net_aux
import torch
import torch.nn as nn
import importlib

class RelDepthModel(nn.Module):
    '''
        Or can use resnext101 for the backbone
    '''
    def __init__(self, backbone='resnet50'):
        super(RelDepthModel, self).__init__()
        encoder = backbone + '_stride32'
        if 'x' in encoder:
            encoder += 'x8d'
        self.depth_model = DepthModel(encoder)

    def predict_depth(self, rgb_image):
        with torch.no_grad():
            input_img = rgb_image.cuda()
            depth = self.depth_model(input_img)
            pred_depth = (depth-depth.min()) + 0.01 # avoids zero depth prediction
            return pred_depth


class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        arr = net_aux.__name__.split('.')
        last = arr[-1]
        encoder_name = last + '.' + encoder
        mod = last
        self.encoder_module = net_aux.resnet50_stride32()
        self.decoder_module = net_aux.Decoder()

    def forward(self, x):
        lateral_out = self.encoder_module(x)
        logit_out = self.decoder_module(lateral_out)
        return logit_out
