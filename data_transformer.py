import torch
from PIL import Image
from torchvision import transforms
from queries_for_download import queries
from pathlib import Path

resize = transforms.Resize((256, 256))  # Resize the image
rotate = transforms.RandomRotation(degrees=(0, 359))  # Random rotation
to_pil = transforms.ToPILImage()  # Convert tensor back to PIL image
to_grayscale = transforms.Grayscale(num_output_channels=1)  # Convert to grayscale (1 channel)

if __name__ == "__main__":
    # create the necessary directories
    output_dir = Path("transformed_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resized_dir = Path("transformed_dataset/resized")
    output_rotated_dir = Path("transformed_dataset/resized_rotated")
    output_resized_dir.mkdir(parents=True, exist_ok=True)
    output_rotated_dir.mkdir(parents=True, exist_ok=True)
    # make train and test directories
    output_resized_train_dir = Path("transformed_dataset/resized/train")
    output_rotated_train_dir = Path("transformed_dataset/resized_rotated/train")
    output_resized_train_dir.mkdir(parents=True, exist_ok=True)
    output_rotated_train_dir.mkdir(parents=True, exist_ok=True)
    output_resized_test_dir = Path("transformed_dataset/resized/test")
    output_rotated_test_dir = Path("transformed_dataset/resized_rotated/test")
    output_resized_test_dir.mkdir(parents=True, exist_ok=True)
    output_rotated_test_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        # make the directory with the query name
        output_resized_test_dir_q = output_resized_test_dir / query
        output_resized_train_dir_q = output_resized_train_dir / query
        output_rotated_test_dir_q = output_rotated_test_dir / query
        output_rotated_train_dir_q = output_rotated_train_dir / query

        output_resized_test_dir_q.mkdir(parents=True, exist_ok=True)
        output_resized_train_dir_q.mkdir(parents=True, exist_ok=True)
        output_rotated_test_dir_q.mkdir(parents=True, exist_ok=True)
        output_rotated_train_dir_q.mkdir(parents=True, exist_ok=True)

        # begin processing images
        for i in range(1, 101):
            try:
                file_name = f"Image_{i}.jpg"
                image_path = Path(f"raw_dataset/{query}/{file_name}")
                image = Image.open(image_path).convert("RGB")  # open and convert Image to RGB with 3 channels
                # resized image for testing
                test_resized_image = resize(image)
                test_resized_image.save(output_resized_test_dir_q / file_name)
                # resized image for training
                train_resized_image = to_grayscale(test_resized_image)
                train_resized_image.save(output_resized_train_dir_q / file_name)
                # rotated image for testing
                test_rotated_image = rotate(test_resized_image)
                test_rotated_image.save(output_rotated_test_dir_q / file_name)
                # rotated image for training
                train_rotated_image = to_grayscale(test_rotated_image)
                train_rotated_image.save(output_rotated_train_dir_q / file_name)

            except FileNotFoundError as e:
                # print(e)
                continue  # some images may not exist since they were deleted in manual post processing, so we skip them
    