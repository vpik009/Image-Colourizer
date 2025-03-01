import torch
import Model_CNN
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# this is where we initialize the model and train it.

to_grayscale = transforms.Compose([ # Convert to grayscale for training data
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
to_pils = transforms.ToPILImage()

if __name__ == "__main__":

    # load train and test data
    dataset = ImageFolder(root="transformed_dataset", transform=transforms.ToTensor()) 

    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # DataLoader makes it easier to deal with batches.
    
    data_iter = iter(data_loader)
    images_label, _ = next(data_iter)  # Unpack images and labels

    images_train = torch.stack([to_grayscale(to_pils(img)) for img in images_label])  # transform to grayscale for training

    # fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # 2 rows, 5 columns

    # for i in range(5):  # Display 5 images
    #     # Original image (RGB)
    #     img_label = images_label[i].permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    #     axes[0, i].imshow(img_label)
    #     axes[0, i].axis("off")
    #     axes[0, i].set_title("Original")

    #     # Grayscale image
    #     img_train = images_train[i].squeeze(0).numpy()  # Remove channel dim
    #     axes[1, i].imshow(img_train, cmap="gray")
    #     axes[1, i].axis("off")
    #     axes[1, i].set_title("Grayscale")

    # plt.tight_layout()
    # plt.show()
    
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