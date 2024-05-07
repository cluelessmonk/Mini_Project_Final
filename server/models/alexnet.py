import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt


async def train_and_get_accuracy(email, no_of_classes):
    script_dir = os.getcwd()
    upload_dir = os.path.join(script_dir, "uploads", email)
    train_dir = os.path.join(upload_dir, "train")
    dev_dir = os.path.join(upload_dir, "dev")
    test_dir = os.path.join(upload_dir, "test")
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

    history = model.fit(
        train_datagen.flow(train_images, labels_categorical, batch_size=batch_size),
        epochs=10
    )

    # Save loss history to a file
    loss_history = history.history['loss']
    with open(os.path.join(upload_dir, "evaluation_results_AN.txt"), 'w') as f:
        f.write("Loss History:\n")
        for loss in loss_history:
            f.write(f"{loss}\n")

    # Save plot of loss function graph
    plt.plot(history.history['loss'])
    plt.title('Alexnet model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig(os.path.join(upload_dir, 'loss_function_plot_AN.png'))
    plt.close()

    # Evaluate the model
    _, train_accuracy = model.evaluate(train_datagen.flow(train_images, labels_categorical, batch_size=batch_size))

    # Save the model
    model.save(os.path.join(upload_dir, "alexnet.keras"))

    # Load the saved model
    model = load_model(os.path.join(upload_dir, "alexnet.keras"))

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

    # Calculate confusion matrices
    dev_conf_matrix = confusion_matrix(dev_labels, dev_predicted_labels)
    test_conf_matrix = confusion_matrix(test_labels, test_predicted_labels)

    # Calculate precision and recall
    dev_precision = precision_score(dev_labels, dev_predicted_labels, average='macro')
    test_precision = precision_score(test_labels, test_predicted_labels, average='macro')

    dev_recall = recall_score(dev_labels, dev_predicted_labels, average='macro')
    test_recall = recall_score(test_labels, test_predicted_labels, average='macro')

    # Classification Reports
    dev_classification_report = classification_report(dev_labels, dev_predicted_labels)
    test_classification_report = classification_report(test_labels, test_predicted_labels)

    # Save evaluation metrics to file
    with open(os.path.join(upload_dir, "evaluation_results_AN.txt"), 'a') as f:
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

    return {"train_accuracy": train_accuracy, "dev_accuracy": dev_accuracy, "test_accuracy": test_accuracy}
