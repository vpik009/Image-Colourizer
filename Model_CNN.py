import torch

class Model_CNN(torch.nn.Module):
    def __init__(self, loss_fn, inputs, labels, device):  #todo to make use of the parameters later
        super(Model_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, padding=2)  # grayscale image, 30, 5x5 kernel
        self.conv2 = torch.nn.Conv2d(in_channels=50, out_channels=150, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=150, out_channels=300, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=300, out_channels=150, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=150, out_channels=20, kernel_size=3, padding=1)
        self.output_layer = torch.nn.Conv2d(20, 3, kernel_size=1)
        
        self.activation = torch.nn.ReLU()
        self.loss = loss_fn
        self.device = device
        self.inputs = inputs
        self.labels = labels
        self.optim = None  # needs to be set after initialization of the model due to model.parameters() argument

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
        
    def train(self, epochs):

        self.to(self.device) # use cuda if available
        super().train(True)

        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(zip(self.inputs, self.labels)):  
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                output = self.forward(inputs)
                self.optim.zero_grad()
                loss = self.loss(output, targets)
                print(loss)
                loss.backward()
                self.optim.step()

            print(f"Train Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.6f}")

        return loss.item()