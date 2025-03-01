import torch

class Model_CNN(torch.nn.Module):
    def __init__(self, loss_fn, train_loader, device):  #todo to make use of the parameters later
        super(Model_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, padding=2)  # grayscale image, 30, 5x5 kernel
        self.conv2 = torch.nn.Conv2d(in_channels=50, out_channels=150, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=150, out_channels=300, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(300, 150, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(50, 20, kernel_size=3, padding=1)
        self.output_layer = torch.nn.Conv2d(20, 3, kernel_size=1)
        
        self.activation = torch.nn.ReLU()
        self.loss = torch.nn.MSELoss()  # L2 loss
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
        
    def train(self, epoch):
        loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):  # train_loader is a DataLoader object and states the batch size
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
        return loss

    def fit(self, epochs):
        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train(epoch)
            val_loss, correct = self.validate()
            if correct > best_acc:
                best_acc = correct
                torch.save(self.model.state_dict(), "best_model.pth")
