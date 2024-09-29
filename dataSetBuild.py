import numpy as np
import os
import pickle
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Initialize lists to hold data, labels, and bounding boxes
data = []
labels = []
bounding_boxes = []  # New list for bounding boxes

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
                    pbar.update(1)
                    continue

                # Resize and normalize the image
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                data.append(img_resized)
                labels.append(class_name)

                # Create dummy bounding boxes (x_min, y_min, x_max, y_max) - modify as needed
                h, w, _ = img.shape
                bounding_boxes.append([0, 0, w, h])  # Full image bounding box

                # Update the progress bar
                pbar.update(1)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)
bounding_boxes = np.array(bounding_boxes)

# Encode labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Verify the unique labels and adjust NUM_CLASSES accordingly
unique_labels = np.unique(labels)
NUM_CLASSES = len(unique_labels)
print(f"Unique labels found: {unique_labels}, NUM_CLASSES set to: {NUM_CLASSES}")

# Shuffle the data
data, labels, bounding_boxes = shuffle(data, labels, bounding_boxes, random_state=0)

# Save the prepared dataset
with open('dataset_for_faster_rcnn.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'bounding_boxes': bounding_boxes}, f)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Bounding boxes shape: {bounding_boxes.shape}")
