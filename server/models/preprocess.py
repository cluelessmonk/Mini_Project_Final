import os
import random
import shutil

async def rename_images_and_split(path, no_of_classes):
    # Iterate through folders
    for class_num in range(1, no_of_classes + 1):
        folder_path = os.path.join(path, str(class_num))
        if os.path.isdir(folder_path):
            # Rename images
            images = os.listdir(folder_path)
            for idx, image in enumerate(images, start=1):
                image_path = os.path.join(folder_path, image)
                new_name = f"{class_num}_{idx}.jpg"  # Change the extension if needed
                new_path = os.path.join(folder_path, new_name)
                os.rename(image_path, new_path)

    # Split into train, dev, and test folders
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
        train_count = int(0.7 * len(images))
        dev_count = test_count = (len(images) - train_count) // 2

        # Move images to train folder
        for image in images[:train_count]:
            src = os.path.join(folder_path, image)
            dest = os.path.join(train_path, image)
            shutil.move(src, dest)

        # Move images to dev folder
        for image in images[train_count:train_count + dev_count]:
            src = os.path.join(folder_path, image)
            dest = os.path.join(dev_path, image)
            shutil.move(src, dest)

        # Move images to test folder
        for image in images[train_count + dev_count:]:
            src = os.path.join(folder_path, image)
            dest = os.path.join(test_path, image)
            shutil.move(src, dest)

        shutil.rmtree(folder_path)

    return True

async def preprocess(email, no_of_classes):
    script_dir = os.getcwd()
    path = os.path.join(script_dir, "uploads")
    path1 = os.path.join(path, email)
    path = os.path.join(path1, "success.txt")
    print(path)
    while True:
        if os.path.exists(path):
            return await rename_images_and_split(path1, no_of_classes)
