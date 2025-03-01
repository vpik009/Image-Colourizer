import torch

class ModelTrain:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, epoch):
        self.model.train()
        loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
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
        print("Best Accuracy: {:.2f}".format(best_acc / len(self.val_loader.dataset) * 100))