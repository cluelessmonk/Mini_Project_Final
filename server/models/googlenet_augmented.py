import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt


async def train_and_get_accuracy(email, no_of_classes):
    script_dir = os.getcwd()
    upload_dir = os.path.join(script_dir, "uploads", email)
    train_dir = os.path.join(upload_dir, "train")
    model_dir = os.path.join(script_dir, "models")
    weights_file = os.path.join(script_dir, "models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
    evaluation_file = os.path.join(upload_dir, "evaluation_results_GN.txt")

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
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

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
    batch_size = 32  # Change batch size to 32

    history = model.fit(
        train_datagen.flow(images, labels_categorical, batch_size=batch_size),
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
    plt.title('GoogleNet Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig(os.path.join(upload_dir, 'loss_function_plot_GN.png'))
    plt.close()

    # Evaluate the model
    _, train_accuracy = model.evaluate(train_datagen.flow(images, labels_categorical, batch_size=32))

    # Save the model
    model.save(os.path.join(upload_dir, "googlenet.keras"))
    dev_dir = os.path.join(upload_dir, "dev")
    test_dir = os.path.join(upload_dir, "test")

    model = load_model(os.path.join(upload_dir, "googlenet.keras"))

    def load_and_preprocess_data(folder):
        images = []
        labels = []
        filenames = []
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(extract_class(filename))
                filenames.append(filename)
            except OSError as e:
                print(f"Error loading image file {filename}: {e}")
                continue  
        return np.array(images), np.array(labels), filenames

    dev_images, dev_labels, _ = load_and_preprocess_data(dev_dir)
    test_images, test_labels, _ = load_and_preprocess_data(test_dir)

    dev_predictions = model.predict(dev_images)
    dev_predicted_labels = np.argmax(dev_predictions, axis=1)

    test_predictions = model.predict(test_images)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    dev_accuracy = np.mean(dev_predicted_labels == dev_labels)
    print("Accuracy on dev set:", dev_accuracy)

    test_accuracy = np.mean(test_predicted_labels == test_labels)
    print("Accuracy on test set:", test_accuracy)

    # Confusion Matrix
    dev_conf_matrix = confusion_matrix(dev_labels, dev_predicted_labels)
    test_conf_matrix = confusion_matrix(test_labels, test_predicted_labels)



    print("Dev Set Confusion Matrix:")
    print(dev_conf_matrix)

    print("Test Set Confusion Matrix:")
    print(test_conf_matrix)

    # Calculate precision and recall
    dev_precision = precision_score(dev_labels, dev_predicted_labels, average='macro')
    test_precision = precision_score(test_labels, test_predicted_labels, average='macro')

    dev_recall = recall_score(dev_labels, dev_predicted_labels, average='macro')
    test_recall = recall_score(test_labels, test_predicted_labels, average='macro')

    # Classification Reports
    dev_classification_report = classification_report(dev_labels, dev_predicted_labels)
    test_classification_report = classification_report(test_labels, test_predicted_labels)

    # Save evaluation metrics to file
    with open(evaluation_file, 'a') as f:
        f.write("\nEvaluation Metrics:\n")
        f.write(f"Train Accuracy: {train_accuracy}\n")
        f.write(f"Dev Accuracy: {dev_accuracy}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write("\nDev Set Confusion Matrix:\n")
        np.savetxt(f, dev_conf_matrix, fmt='%d')
        f.write("\nTest Set Confusion Matrix:\n")
        np.savetxt(f, test_conf_matrix, fmt='%d')
        f.write(f"\nDev Precision: {dev_precision}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"\nDev Recall: {dev_recall}\n")
        f.write(f"Test Recall: {test_recall}\n")
        f.write("\nClassification Report (Dev Set):\n")
        f.write(dev_classification_report)
        f.write("\nClassification Report (Test Set):\n")
        f.write(test_classification_report)

    return {
        "train_accuracy": train_accuracy, 
        "dev_accuracy": dev_accuracy, 
        "test_accuracy": test_accuracy,
        "train_loss": history.history['loss'][-1]
    }
