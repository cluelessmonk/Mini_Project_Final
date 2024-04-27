async def train_and_get_accuracy(email, no_of_classes):
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array, load_img
    from PIL import Image
    
    script_dir = os.getcwd()
    train_dir = os.path.join(script_dir, "uploads", email, "train")
    model_dir = os.path.join(script_dir, "models")
    weights_file = os.path.join(script_dir, "models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    input_shape = (224, 224, 3)
    num_classes = no_of_classes

    def gray_to_rgb(img):
        if img.ndim == 2:
            img_rgb = np.stack((img,) * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = np.concatenate((img, img, img), axis=-1)
        elif img.ndim == 3 and img.shape[2] == 3:
            img_rgb = img
        else:
            raise ValueError("Unsupported image shape")
        return img_rgb

    def extract_class(filename):
        # Assuming filenames start with class labels (e.g., '1_', '2_', etc.)
        class_label = int(filename.split('_')[0])
        return class_label - 1 

    train_datagen = ImageDataGenerator(
        preprocessing_function=gray_to_rgb,
        rescale=1./255
    )

    def load_data():
        images = []
        labels = []
        for filename in os.listdir(train_dir):
                try:
                    img = Image.open(os.path.join(train_dir, filename))
                    img = img.resize((input_shape[0], input_shape[1]))
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(extract_class(filename))
                except OSError as e:
                    print(f"Error loading image file {filename}: {e}")
        return np.array(images), np.array(labels)

    base_model = InceptionV3(weights=weights_file, include_top=False, input_shape=input_shape)
    predictions = Dense(num_classes, activation='softmax')(GlobalAveragePooling2D()(base_model.output))

    # Load the data
    images, labels = load_data()
    labels_categorical = np.eye(num_classes)[labels]

    # Combine base model and layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze layers of base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    batch_size = 32  # Changed batch size here
    steps_per_epoch = len(images) // batch_size

    model.fit(
        train_datagen.flow(images, labels_categorical, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=10
    )

    # Evaluate the model
    _, train_accuracy = model.evaluate(train_datagen.flow(images, labels_categorical, batch_size=32))

    # Save the model
    model.save(os.path.join(script_dir, "uploads", email, "googlenet.keras"))
    dev_dir = os.path.join(script_dir, "uploads", email, "dev")
    test_dir = os.path.join(script_dir, "uploads", email, "test")

    model = load_model(os.path.join(script_dir, "uploads", email, "googlenet.keras"))
    def load_and_preprocess_data(folder):
        images = []
        labels = []
        filenames = []
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)
                img = img.resize((224, 224))  # Resize the image
                img_array = np.array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(extract_class(filename))
                filenames.append(filename)
            except OSError as e:
                print(f"Error loading image file {filename}: {e}")
                continue  # Skip over the corrupted image
        return np.array(images), np.array(labels), filenames


    dev_images, dev_labels, dev_filenames = load_and_preprocess_data(dev_dir)

    test_images, test_labels, test_filenames = load_and_preprocess_data(test_dir)

    dev_predictions = model.predict(dev_images)
    dev_predicted_labels = np.argmax(dev_predictions, axis=1)

    test_predictions = model.predict(test_images)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    dev_accuracy = np.mean(dev_predicted_labels == dev_labels)
    print("Accuracy on dev set:", dev_accuracy)

    test_accuracy = np.mean(test_predicted_labels == test_labels)
    print("Accuracy on test set:", test_accuracy)
    return {"train_accuracy":train_accuracy,"dev_accuracy": dev_accuracy,"test_accuracy": test_accuracy}
