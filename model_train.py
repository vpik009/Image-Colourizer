import torch
from Model_CNN import Model_CNN
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

MODEL_PATH = "model.pth"  # latest model path
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
    dataset = ImageFolder(root="transformed_dataset", transform=transforms.ToTensor()) 

    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # DataLoader makes it easier to deal with batches.
    data_iter = iter(data_loader)
    images_label, _ = next(data_iter)  # Unpack images and labels

    images_train = torch.stack([to_grayscale(to_pils(img)) for img in images_label])  # transform to grayscale for training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # I only have cpu
    loss_fn = torch.nn.L1Loss() #torch.nn.MSELoss()

    model = Model_CNN(loss_fn, device)  # Initialize without optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.optim = optimizer  # Attach optimizer to model

    # load existing model (if any)
    # model.load_state_dict(torch.load("model.pth", map_location=device))
    start_epoch = 0
    start_batch = 0
    try:
        checkpoint_file = "model_epoch_0_batch_4.pth"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load model and optimizer states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Get the epoch and batch number to continue training
        start_batch = checkpoint.get("batch", 0)
        start_epoch = checkpoint["epoch"]
        print(f"Loaded model from epoch {checkpoint['epoch']}, batch {checkpoint.get('batch', 0)}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")


    # train the model
    print(f"Number of batches: {len(data_loader)}")
    epochs = 10
    model.train()
    print("\nTraining the model...")
    for i in range(start_epoch, epochs):
        print(f"Epoch {i}")

        for batch_idx, (targets, _) in enumerate(data_loader):  # targets = colored images

            if batch_idx < start_batch:  # Skip processed batches if loading from checkpoint
                continue

            inputs = torch.stack([to_grayscale(to_pils(img)) for img in targets])  # get out train data
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            output = model(inputs)
            model.optim.zero_grad()
            loss = model.regularization_loss(output, targets, lambda_l2=0.001)  # calculate loss with regularization
            print(f"Batch {batch_idx} - Loss: {loss.item():.6f}")  # Print loss
            loss.backward()
            model.optim.step()
            
            # save the model every x batches
            if (batch_idx + 1) % 5 == 0:
                batch_save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{i}_batch_{batch_idx}.pth")
                torch.save({
                    "epoch": i,
                    "batch": batch_idx + 1,  # start from next batch
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, batch_save_path)
                print(f"Saved batch checkpoint: {batch_save_path}")
        
        # save the model every epoch
        epoch_save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{i}.pth")
        torch.save({
            "epoch": i + 1,  # save next epoch to start from
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, epoch_save_path)
        print(f"Saved epoch checkpoint: {epoch_save_path}")

    # test the model
    d = '/home/vladislav/Documents/Image-Colourizer/transformed_dataset/resized/animals/Image_25.jpg'
    t_1 = "test_image1.jpg"
    t_2 = "test_image2.jpg"
    t_3 = "test_image2_small.jpg"
    image = Image.open(d)
    image = to_grayscale(image).unsqueeze(0)
    output = model.forward(image)

    # Display the image
    output_image = output.squeeze(0).cpu().detach()  # Remove batch dim & move to CPU
    output_image = to_pils(output_image)
    plt.imshow(output_image)
    plt.axis("off")  # Hide axes
    plt.show()

