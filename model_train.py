import torch
import Model_CNN
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# this is where we initialize the model and train it.

toGrayscale = transforms.Compose([ # Convert to grayscale for training data
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

if __name__ == "__main__":

    dataset = ImageFolder(root="transformed_dataset", transform=toGrayscale)  # Load dataset with transform
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # Unpack images and labels

    print(images)
    # Convert tensor to PIL image (for visualization)
    to_pil = transforms.ToPILImage()

    fig, axes = plt.subplots(1, 5, figsize=(12, 6))  # Show 5 images
    for i in range(5):
        img_tensor = images[i].squeeze(0)  # Remove channel dim (C, H, W) -> (H, W)
        img = to_pil(img_tensor)  # Convert to PIL image
        axes[i].imshow(img, cmap="gray")  # Display as grayscale
        axes[i].axis("off")

    plt.show()
    raise ValueError
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # I only have cpu
    loss_fn = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader([], batch_size=20)  # use a dummy dataloader that uses a batch size
    model = Model_CNN(loss_fn, None, train_loader, device)  # Initialize without optimizer

    # Define optimizer AFTER model initialization (important!)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.optimizer = optimizer  # Attach optimizer to model

    # train the model

    # fit the model

    # test the model

    # save the model