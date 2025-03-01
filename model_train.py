import torch
from Model_CNN import Model_CNN
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
to_tensor = transforms.ToTensor()

if __name__ == "__main__":
    
    # load train and test data
    dataset = ImageFolder(root="transformed_dataset", transform=transforms.ToTensor()) 

    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # DataLoader makes it easier to deal with batches.
    print(f"Number of batches: {len(data_loader)}")
    data_iter = iter(data_loader)
    images_label, _ = next(data_iter)  # Unpack images and labels

    images_train = torch.stack([to_grayscale(to_pils(img)) for img in images_label])  # transform to grayscale for training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # I only have cpu
    loss_fn = torch.nn.L1Loss() #torch.nn.MSELoss()

    model = Model_CNN(loss_fn, device)  # Initialize without optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.optim = optimizer  # Attach optimizer to model

    # load existing model (if have)
    # model.load_state_dict(torch.load("model.pth", map_location=device))

    # train the model
    epochs = 10
    print("\nTraining the model...")
    model.train()
    for i in range(epochs):
        print(f"Epoch {i+1}")

        for batch_idx, (targets, _) in enumerate(data_loader):  # targets = colored images
            inputs = torch.stack([to_grayscale(to_pils(img)) for img in targets])  # get out train data
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            output = model(inputs)
            model.optim.zero_grad()
            loss = model.loss(output, targets)
            print(f"Batch {batch_idx + 1} - Loss: {loss.item():.6f}")  # Print loss
            
            loss.backward()
            model.optim.step()
            #break  # testing

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # test the model
    d = '/home/vladislav/Documents/Image-Colourizer/transformed_dataset/resized/animals/Image_20.jpg'
    t = "test_image1.jpg"
    image = Image.open(d)
    image = to_grayscale(image).unsqueeze(0)
    output = model.forward(image)

    # Display the image
    output_image = output.squeeze(0).cpu().detach()  # Remove batch dim & move to CPU
    output_image = to_pils(output_image)
    plt.imshow(output_image)
    plt.axis("off")  # Hide axes
    plt.show()

