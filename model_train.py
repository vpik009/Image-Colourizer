import torch
import Model_CNN
# this is where we initialize the model and train it.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # I only have cpu
loss_fn = torch.nn.MSELoss()
train_loader = torch.utils.data.DataLoader([], batch_size=20)  # use a dummy dataloader that uses a batch size
model = Model_CNN(loss_fn, None, train_loader, device)  # Initialize without optimizer

# Define optimizer AFTER model initialization (important!)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.optimizer = optimizer  # Attach optimizer to model