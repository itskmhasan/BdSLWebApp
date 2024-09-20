import numpy as np
import os
import pickle
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tqdm import tqdm  # Importing tqdm for progress bar

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Initialize lists to hold data and labels
data = []
labels = []

# Directory containing the dataset
dataset_dir = 'sign_language_dataset'

# Load the images and labels
class_dirs = [os.path.join(dataset_dir, class_name) for class_name in os.listdir(dataset_dir) if
              os.path.isdir(os.path.join(dataset_dir, class_name))]
total_images = sum(len(os.listdir(class_dir)) for class_dir in class_dirs)

# Using tqdm to show progress bar
with tqdm(total=total_images, desc="Building Dataset", unit="image") as pbar:
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Read the image
                img = cv2.imread(img_path)

                # Check if the image was loaded successfully
                if img is None:
                    print(f"Warning: Unable to load image at {img_path}. Skipping.")
                    pbar.update(1)  # Update progress bar for skipped image
                    continue

                # Resize and normalize the image
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                data.append(img_resized)
                labels.append(class_name)

                # Update the progress bar
                pbar.update(1)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Verify the unique labels and adjust NUM_CLASSES accordingly
unique_labels = np.unique(labels)
NUM_CLASSES = len(unique_labels)
print(f"Unique labels found: {unique_labels}, NUM_CLASSES set to: {NUM_CLASSES}")

# Ensure labels are within the valid range [0, NUM_CLASSES-1]
if np.any(labels >= NUM_CLASSES):
    raise ValueError(f"Labels contain values outside the range [0, {NUM_CLASSES - 1}]")

# One-hot encode labels
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Shuffle the data
data, labels = shuffle(data, labels, random_state=0)

# Save the prepared dataset
with open('dataset_for_cnn.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
