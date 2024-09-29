import numpy as np
import pickle
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

# Load and prepare data
with open('dataset_for_cnn.pickle', 'rb') as f:
    data_dict = pickle.load(f)
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Ensure NUM_CLASSES is accurate
NUM_CLASSES = labels.shape[1]

# Split data into training and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2, random_state=42)


# Load a pre-trained Faster R-CNN model from TensorFlow Hub
def load_model(model_url):
    model = hub.load(model_url).signatures['serving_default']
    return model


# Define the correct model URL
model_url = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
model = load_model(model_url)


# Prepare for predictions (for demonstration)
def predict(model, images):
    # Preprocess images as needed by Faster R-CNN (resize and normalization)
    images_resized = [cv2.resize(image, (640, 640)) for image in images]  # Change size as needed
    images_tensor = tf.convert_to_tensor(images_resized)

    # Run inference
    output_dict = model(images_tensor)
    return output_dict


# Example of how to use the model for predictions
# Replace this part with your actual training loop if needed
predictions_train = predict(model, data_train)
predictions_val = predict(model, data_val)

print("Model predictions completed.")