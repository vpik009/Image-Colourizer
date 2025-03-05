import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_CNN(nn.Module):
    def __init__(self, loss_fn, device):
        super(Model_CNN, self).__init__()

        # encoder layers
        self.enc1 = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)  # reduce spatial size by 2
        
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)  # reduce spatial size by 2

        # bottlenecking layer
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        # decoder layers
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # increase spatial size by 2
        self.dec1 = nn.Sequential(nn.Conv2d(64, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(48, 24, 2, stride=2)  # increase spatial size by 2
        self.dec2 = nn.Sequential(nn.Conv2d(24, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU())

        # output layers
        self.output_layer = nn.Conv2d(12, 3, 1)
        self.output_activation = nn.Sigmoid()  # each rgb pixel range is [0, 1]

        # other params
        self.loss = loss_fn
        self.device = device
        self.optim = None

    def regularization_loss(self, input, target, lambda_l2):
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        return self.loss(input, target) + lambda_l2 * l2_reg

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.pool1(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.pool2(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x_out = self.output_layer(x)
        x_out = self.output_activation(x_out)
        return x_out
