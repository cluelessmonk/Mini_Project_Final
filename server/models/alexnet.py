async def train_and_get_accuracy(email, no_of_classes):
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array, load_img
    from PIL import Image
    
    script_dir = os.getcwd()
    train_dir = os.path.join(script_dir, "uploads", email, "train")
    model_dir = os.path.join(script_dir, "models")

    input_shape = (227, 227, 3)  # AlexNet input shape
    num_classes = no_of_classes

    def extract_class(filename):
        # Assuming filenames start with class labels (e.g., '1_', '2_', etc.)
        class_label = int(filename.split('_')[0])
        return class_label - 1 

    train_datagen = ImageDataGenerator(rescale=1./255)

    def load_data(directory):
        images = []
        labels = []
        for filename in os.listdir(directory):
            try:
                img = Image.open(os.path.join(directory, filename))
                img = img.resize((input_shape[0], input_shape[1]))
                img_array = np.array(img)
                images.append(img_array)
                labels.append(extract_class(filename))
            except OSError as e:
                print(f"Error loading image file {filename}: {e}")
        return np.array(images), np.array(labels)

    # Define AlexNet model
    model = Sequential([
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the data
    train_images, train_labels = load_data(train_dir)
    labels_categorical = np.eye(num_classes)[train_labels]

    # Train the model
    batch_size = 32  # Change batch size to 32
    steps_per_epoch = len(train_images) // batch_size

    model.fit(
        train_datagen.flow(train_images, labels_categorical, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=10
    )

    # Evaluate the model
    _, train_accuracy = model.evaluate(train_datagen.flow(train_images, labels_categorical, batch_size=batch_size))

    # Save the model
    model.save(os.path.join(script_dir, "uploads", email, "alexnet.keras"))

    # Load the saved model
    model = load_model(os.path.join(script_dir, "uploads", email, "alexnet.keras"))

    # Load and preprocess data for dev and test sets
    def load_and_preprocess_data(folder):
        images = []
        labels = []
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = load_img(img_path, target_size=(227, 227))
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(extract_class(filename))
            except Exception as e:
                print(f"Error loading image file {filename}: {e}")
                continue  # Skip over the corrupted image
        return np.array(images), np.array(labels)


    dev_dir = os.path.join(script_dir, "uploads", email, "dev")
    test_dir = os.path.join(script_dir, "uploads", email, "test")

    dev_images, dev_labels = load_and_preprocess_data(dev_dir)
    test_images, test_labels = load_and_preprocess_data(test_dir)

    # Predictions for dev and test sets
    dev_predictions = model.predict(dev_images)
    dev_predicted_labels = np.argmax(dev_predictions, axis=1)

    test_predictions = model.predict(test_images)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    # Calculate accuracy for dev and test sets
    dev_accuracy = np.mean(dev_predicted_labels == dev_labels)
    print("Accuracy on dev set:", dev_accuracy)

    test_accuracy = np.mean(test_predicted_labels == test_labels)
    print("Accuracy on test set:", test_accuracy)

    return {"train_accuracy": train_accuracy, "dev_accuracy": dev_accuracy, "test_accuracy": test_accuracy}
