from PIL import Image
from torchvision import transforms
from queries_for_download import queries
from pathlib import Path

resize = transforms.Resize((256, 256))  # Resize the image
rotate = transforms.RandomRotation(degrees=(0, 359))  # Random rotation
to_pil = transforms.ToPILImage()  # Convert tensor back to PIL image

if __name__ == "__main__":
    # create the necessary directories
    print("Transforming data...")
    output_dir = Path("transformed_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resized_dir = Path("transformed_dataset/resized")
    output_rotated_dir = Path("transformed_dataset/resized_rotated")
    output_resized_dir.mkdir(parents=True, exist_ok=True)
    output_rotated_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        # make the directory with the query name
        resized_query_dir = output_resized_dir / query
        rotated_query_dir = output_rotated_dir / query
        resized_query_dir.mkdir(parents=True, exist_ok=True)
        rotated_query_dir.mkdir(parents=True, exist_ok=True)

        # begin processing images
        for i in range(1, 101):
            try:
                file_name = f"Image_{i}.jpg"
                image_path = Path(f"raw_dataset/{query}/{file_name}")
                image = Image.open(image_path).convert("RGB")  # open and convert Image to RGB with 3 channels
                resized_image = resize(image)
                resized_image.save(output_resized_dir / query / file_name)
                rotated_image = rotate(resized_image)
                rotated_image.save(output_rotated_dir / query / file_name)
            except FileNotFoundError as e:
                continue  # some images may not exist since they were deleted in manual post processing, so we skip them
    print("Data transformation complete")
    