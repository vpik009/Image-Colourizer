import torch

class Model_CNN(torch.nn.Module):
    def __init__(self, loss_fn, device):
        super(Model_CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv7 = torch.nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
        self.conv10 = torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv11 = torch.nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1)
        self.conv12 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv13 = torch.nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, padding=1)
        self.conv14 = torch.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.output_layer = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

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

        self.activation = torch.nn.ReLU()
        self.output_activation = torch.nn.Sigmoid()
        self.loss = loss_fn
        self.device = device
        self.optim = None  # Needs to be set after model initialization

    def regularization_loss(self, input, target, lambda_l2):
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        return self.loss(input, target) + lambda_l2 * l2_reg

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.activation(self.bn7(self.conv7(x)))
        x = self.activation(self.bn8(self.conv8(x)))
        x = self.activation(self.bn9(self.conv9(x)))
        x = self.activation(self.bn10(self.conv10(x)))
        x = self.activation(self.bn11(self.conv11(x)))
        x = self.activation(self.bn12(self.conv12(x)))
        x = self.activation(self.bn13(self.conv13(x)))
        x = self.activation(self.bn14(self.conv14(x)))
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x
