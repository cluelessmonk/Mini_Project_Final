async def train_and_get_accuracy_resnet50(email, no_of_classes):
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from keras.preprocessing.image import img_to_array, load_img
    from PIL import Image
    
    script_dir = os.getcwd()
    train_dir = os.path.join(script_dir, "uploads", email, "train")
    dev_dir = os.path.join(script_dir, "uploads", email, "dev")
    test_dir = os.path.join(script_dir, "uploads", email, "test")
    weights_path = os.path.join(script_dir, "models", "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
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
        class_label = int(filename.split('_')[0])
        return class_label - 1 

    train_datagen = ImageDataGenerator(
        preprocessing_function=gray_to_rgb,
        rescale=1./255
    )

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

    base_model = ResNet50(weights=weights_path, include_top=False, input_shape=input_shape)
    predictions = Dense(num_classes, activation='softmax')(GlobalAveragePooling2D()(base_model.output))

    train_images, train_labels = load_data(train_dir)
    dev_images, dev_labels = load_data(dev_dir)
    test_images, test_labels = load_data(test_dir)

    train_labels_categorical = np.eye(num_classes)[train_labels]
    dev_labels_categorical = np.eye(num_classes)[dev_labels]
    test_labels_categorical = np.eye(num_classes)[test_labels]

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 32
    steps_per_epoch = len(train_images) // batch_size

    model.fit(
        train_datagen.flow(train_images, train_labels_categorical, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=10
    )

    _, train_accuracy = model.evaluate(train_datagen.flow(train_images, train_labels_categorical, batch_size=batch_size))
    _, dev_accuracy = model.evaluate(train_datagen.flow(dev_images, dev_labels_categorical, batch_size=batch_size))
    _, test_accuracy = model.evaluate(train_datagen.flow(test_images, test_labels_categorical, batch_size=batch_size))

    model.save(os.path.join(script_dir, "uploads", email, "resnet50.keras"))

    return {"train_accuracy": train_accuracy, "dev_accuracy": dev_accuracy, "test_accuracy": test_accuracy}
