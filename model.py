import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, padding=2)  # grayscale image, 30, 5x5 kernel
        self.conv2 = torch.nn.Conv2d(in_channels=50, out_channels=150, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=150, out_channels=300, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(300, 150, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(50, 20, kernel_size=3, padding=1)
        self.output_layer = torch.nn.Conv2d(20, 3, kernel_size=1)
        
        self.activation = torch.nn.ReLU()
        self.loss = torch.nn.MSELoss()  # L2 loss

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x
        
