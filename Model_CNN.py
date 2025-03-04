import torch

class Model_CNN(torch.nn.Module):
    def __init__(self, loss_fn, device):
        super(Model_CNN, self).__init__()

        # Define convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 12, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(12, 16, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 24, 5, padding=2)
        self.conv4 = torch.nn.Conv2d(24, 32, 5, padding=2)
        self.conv5 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.conv6 = torch.nn.Conv2d(64, 32, 5, padding=2)
        self.conv7 = torch.nn.Conv2d(32, 24, 3, padding=1)
        self.conv8 = torch.nn.Conv2d(24, 16, 3, padding=1)
        self.conv9 = torch.nn.Conv2d(16, 12, 3, padding=1)
        self.conv10 = torch.nn.Conv2d(12, 12, 3, padding=1)
        self.conv11 = torch.nn.Conv2d(12, 8, 3, padding=1)
        self.conv12 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.conv13 = torch.nn.Conv2d(8, 6, 3, padding=1)
        self.conv14 = torch.nn.Conv2d(6, 6, 3, padding=1)
        self.output_layer = torch.nn.Conv2d(6, 3, 1)

        # Define batch normalization layers
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(24)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.bn6 = torch.nn.BatchNorm2d(32)
        self.bn7 = torch.nn.BatchNorm2d(24)
        self.bn8 = torch.nn.BatchNorm2d(16)
        self.bn9 = torch.nn.BatchNorm2d(12)
        self.bn10 = torch.nn.BatchNorm2d(12)
        self.bn11 = torch.nn.BatchNorm2d(8)
        self.bn12 = torch.nn.BatchNorm2d(8)
        self.bn13 = torch.nn.BatchNorm2d(6)
        self.bn14 = torch.nn.BatchNorm2d(6)

        # Define skip connections (1x1 convolutions to match channels)
        self.skip1 = torch.nn.Conv2d(12, 16, 1)
        self.skip2 = torch.nn.Conv2d(16, 24, 1)
        self.skip3 = torch.nn.Conv2d(24, 32, 1)
        self.skip4 = torch.nn.Conv2d(32, 64, 1)
        self.skip5 = torch.nn.Conv2d(64, 32, 1)
        self.skip6 = torch.nn.Conv2d(32, 24, 1)
        self.skip7 = torch.nn.Conv2d(24, 16, 1)
        self.skip8 = torch.nn.Conv2d(16, 12, 1)

        # Define activation functions
        self.activation = torch.nn.ReLU()
        self.output_activation = torch.nn.Sigmoid()

        # Loss function and device
        self.loss = loss_fn
        self.device = device
        self.optim = None

    def regularization_loss(self, input, target, lambda_l2):
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        return self.loss(input, target) + lambda_l2 * l2_reg

    def forward(self, x):
        x1 = self.activation(self.bn1(self.conv1(x)))
        x2 = self.activation(self.bn2(self.conv2(x1))) + self.skip1(x1)
        x3 = self.activation(self.bn3(self.conv3(x2))) + self.skip2(x2)
        x4 = self.activation(self.bn4(self.conv4(x3))) + self.skip3(x3)
        x5 = self.activation(self.bn5(self.conv5(x4))) + self.skip4(x4)
        x6 = self.activation(self.bn6(self.conv6(x5))) + self.skip5(x5)
        x7 = self.activation(self.bn7(self.conv7(x6))) + self.skip6(x6)
        x8 = self.activation(self.bn8(self.conv8(x7))) + self.skip7(x7)
        x9 = self.activation(self.bn9(self.conv9(x8))) + self.skip8(x8)
        x10 = self.activation(self.bn10(self.conv10(x9)))
        x11 = self.activation(self.bn11(self.conv11(x10)))
        x12 = self.activation(self.bn12(self.conv12(x11)))
        x13 = self.activation(self.bn13(self.conv13(x12)))
        x14 = self.activation(self.bn14(self.conv14(x13)))
        x_out = self.output_layer(x14)
        x_out = self.output_activation(x_out)
        return x_out