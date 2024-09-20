import numpy as np
import pickle
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Load and prepare data
with open('dataset_for_cnn.pickle', 'rb') as f:
    data_dict = pickle.load(f)
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Ensure NUM_CLASSES is accurate
NUM_CLASSES = labels.shape[1]

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.2)

# Save the trained model in the recommended Keras format
model.save('sign_language_cnn_model.keras')

print("Model training completed and saved as 'sign_language_cnn_model.keras'.")