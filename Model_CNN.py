import torch

class Model_CNN(torch.nn.Module):
    def __init__(self, loss_fn, device):  #todo to make use of the parameters later
        super(Model_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, padding=2)  # grayscale image, 30, 5x5 kernel
        self.conv2 = torch.nn.Conv2d(in_channels=50, out_channels=150, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=150, out_channels=300, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=300, out_channels=150, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=150, out_channels=20, kernel_size=3, padding=1)
        self.output_layer = torch.nn.Conv2d(20, 3, kernel_size=1)

        # prevent vanishing gradient
        self.bn1 = torch.nn.BatchNorm2d(50)
        self.bn2 = torch.nn.BatchNorm2d(150)
        self.bn3 = torch.nn.BatchNorm2d(300)
        self.bn4 = torch.nn.BatchNorm2d(150)
        self.bn5 = torch.nn.BatchNorm2d(20)

        
        self.activation = torch.nn.ReLU()
        self.loss = loss_fn
        self.device = device
        self.optim = None  # needs to be set after initialization of the model due to model.parameters() argument

    def regularization_loss(self, input, target, lambda_l2):
        l2_reg = 0
        loss = self.loss(input, target)
        for param in self.parameters():
            l2_reg += param.pow(2).sum()
        loss += lambda_l2 * l2_reg
        return loss

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.output_layer(x)
        return x