import os
import random
import shutil

async def rename_images_and_split(path, no_of_classes):
    # Rename images and split into train, dev, and test folders
    train_path = os.path.join(path, "train")
    dev_path = os.path.join(path, "dev")
    test_path = os.path.join(path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(dev_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for class_num in range(1, no_of_classes + 1):
        folder_path = os.path.join(path, str(class_num))
        images = os.listdir(folder_path)
        random.shuffle(images)
        total_count = len(images)
        train_count = int(0.7 * total_count)
        dev_count = test_count = (total_count - train_count) // 2

        # Rename and move images to train folder
        for idx, image in enumerate(images[:train_count], start=1):
            old_image_path = os.path.join(folder_path, image)
            new_image_name = f"{class_num}_{idx}.jpg"  # Change extension if needed
            new_image_path = os.path.join(train_path, new_image_name)
            shutil.move(old_image_path, new_image_path)

        # Rename and move images to dev folder
        for idx, image in enumerate(images[train_count:train_count + dev_count], start=1):
            old_image_path = os.path.join(folder_path, image)
            new_image_name = f"{class_num}_{train_count + idx}.jpg"  # Change extension if needed
            new_image_path = os.path.join(dev_path, new_image_name)
            shutil.move(old_image_path, new_image_path)

        # Rename and move images to test folder
        for idx, image in enumerate(images[train_count + dev_count:], start=1):
            old_image_path = os.path.join(folder_path, image)
            new_image_name = f"{class_num}_{train_count + dev_count + idx}.jpg"  # Change extension if needed
            new_image_path = os.path.join(test_path, new_image_name)
            shutil.move(old_image_path, new_image_path)

        shutil.rmtree(folder_path)

    return True

async def preprocess(email, no_of_classes):
    script_dir = os.getcwd()
    path = os.path.join(script_dir, "uploads", email)
    await rename_images_and_split(path, no_of_classes)
