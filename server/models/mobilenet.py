import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

async def train_and_get_accuracy_mobilenet(email, no_of_classes):
    script_dir = os.getcwd()
    upload_dir = os.path.join(script_dir, "uploads", email)
    train_dir = os.path.join(upload_dir, "train")
    dev_dir = os.path.join(upload_dir, "dev")
    test_dir = os.path.join(upload_dir, "test")
    weights_path = os.path.join(script_dir, "models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    evaluation_file = os.path.join(upload_dir, "evaluation_results.txt")
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

    base_model = MobileNetV2(weights=weights_path, include_top=False, input_shape=input_shape)
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

    batch_size = 32  # Changed batch size here

    history = model.fit(
        train_datagen.flow(train_images, train_labels_categorical, batch_size=batch_size),
        steps_per_epoch=len(train_images)//batch_size,
        epochs=10
    )

    # Save loss history to a file
    loss_history = history.history['loss']
    with open(evaluation_file, 'w') as f:
        f.write("Loss History:\n")
        for loss in loss_history:
            f.write(f"{loss}\n")

    # Save plot of loss function graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig(os.path.join(upload_dir, 'loss_function_plot_MN.png'))
    plt.close()

    _, train_accuracy = model.evaluate(train_datagen.flow(train_images, train_labels_categorical, batch_size=batch_size))
    _, dev_accuracy = model.evaluate(train_datagen.flow(dev_images, dev_labels_categorical, batch_size=batch_size))
    _, test_accuracy = model.evaluate(train_datagen.flow(test_images, test_labels_categorical, batch_size=batch_size))

    # Predict labels for dev and test sets
    dev_predictions = model.predict(dev_images)
    test_predictions = model.predict(test_images)

    # Calculate confusion matrices
    dev_conf_matrix = confusion_matrix(dev_labels, np.argmax(dev_predictions, axis=1))
    test_conf_matrix = confusion_matrix(test_labels, np.argmax(test_predictions, axis=1))

    # Calculate precision and recall
    dev_precision = precision_score(dev_labels, np.argmax(dev_predictions, axis=1), average='macro')
    test_precision = precision_score(test_labels, np.argmax(test_predictions, axis=1), average='macro')

    dev_recall = recall_score(dev_labels, np.argmax(dev_predictions, axis=1), average='macro')
    test_recall = recall_score(test_labels, np.argmax(test_predictions, axis=1), average='macro')

    # Save evaluation metrics to file
    with open(evaluation_file, 'a') as f:
        f.write("\nConfusion Matrix (Dev Set):\n")
        np.savetxt(f, dev_conf_matrix, fmt='%d')
        f.write("\nConfusion Matrix (Test Set):\n")
        np.savetxt(f, test_conf_matrix, fmt='%d')
        f.write(f"\nDev Precision: {dev_precision}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"\nDev Recall: {dev_recall}\n")
        f.write(f"Test Recall: {test_recall}\n")

    model.save(os.path.join(upload_dir, "mobilenetV2.keras"))

    return {"train_accuracy": train_accuracy, "dev_accuracy": dev_accuracy, "test_accuracy": test_accuracy}



