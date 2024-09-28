import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

import os

dataset_path = r"E:\Hasan\BdSLWebApp\sign_language_dataset"

# Define the dimensions of your images
img_width, img_height = 64, 64

# Initialize lists to hold the data and labels
data = []
labels = []

# Loop over the dataset directory
for i, class_folder in enumerate(os.listdir(dataset_path)):
    class_folder_path = os.path.join(dataset_path, class_folder)

    # Check if the path is a directory (to skip files like .gitignore)
    if os.path.isdir(class_folder_path):
        # Loop over the images in each class folder
        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            # Load the image, resize to the desired dimensions, and convert to gray scale
            image = cv2.imread(image_path)
            image = cv2.resize(image, (img_width, img_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Add the pre-processed image and its label to the lists
            data.append(image)
            labels.append(i)

# Convert the data and labels lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

# Normalize the pixel values to be between 0 and 1
train_data = train_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# Convert labels to one-hot encoded vectors
num_classes = len(np.unique(labels))
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=train_data.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# Train the model
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))

# Save the trained model
model.save("sign_language_model.h5")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Calculate confusion matrix and classification report
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print("Confusion matrix:\n", conf_matrix)
print("Classification report:\n", class_report)

# Compute sensitivity and specificity
for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp
    tn = np.sum(conf_matrix) - tp - fp - fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Class {i}: Sensitivity = {sensitivity:.2f}, Specificity = {specificity:.2f}")
