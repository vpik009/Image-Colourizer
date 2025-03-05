import torch
from Model_CNN import Model_CNN
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # make checkpoint dir if doesnt already exist
to_grayscale = transforms.Compose([ # Convert to grayscale for training data
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
to_pils = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

if __name__ == "__main__":
    # load train and test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # I only have cpu
    print(device)
    loss_fn = torch.nn.MSELoss()

    model = Model_CNN(loss_fn, device)  # Initialize without optimizer
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model.optim = optimizer  # Attach optimizer to model

    # load existing model
    start_epoch = 0
    start_batch = 0
    checkpoint_file = "model_epoch_90.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Get the epoch and batch number to continue training
    start_batch = checkpoint.get("batch", 0)
    start_epoch = checkpoint["epoch"]
    print(f"Loaded model from epoch {checkpoint['epoch']}, batch {checkpoint.get('batch', 0)}")

    # d = '/home/vladislav/Documents/Image-Colourizer/transformed_dataset/resized/animals/Image_22.jpg'
    # p = '/home/vladislav/Documents/Image-Colourizer/transformed_dataset/resized/people/Image_24.jpg'
    # f = '/home/vladislav/Documents/Image-Colourizer/transformed_dataset/resized_rotated/food/Image_17.jpg'
    t_1 = "test_image1.jpg"
    t_1_small = "test_image1_small.jpg"
    t_2 = "test_image2.jpg"
    t_2_small = "test_image2_small.jpg"
    t_4 = "test_image3.jpg"
    image = Image.open(t_1_small)
    image = to_grayscale(image).unsqueeze(0)
    image = image.to(device)  # allow for GPU processing
    output = model.forward(image)

    # Display the image
    output_image = output.squeeze(0).cpu().detach()  # Remove batch dim & move to CPU
    output_image = to_pils(output_image)
    plt.imshow(output_image)
    plt.axis("off")  # Hide axes
    plt.show()