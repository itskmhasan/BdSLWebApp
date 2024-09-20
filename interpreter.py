import warnings
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Suppress warnings related to protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the trained model
trainedModel = tf.keras.models.load_model('sign_language_cnn_model.keras')

# Starting the webcam
cap = cv2.VideoCapture(0)

# Initializing the hand landmarks detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Path to the Bengali font
font_path = './SolaimanLipi_22-02-2012.ttf'  # Ensure this path is correct
font_size = 60
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"Could not open font file at {font_path}. Please check the path and try again.")
    exit()

def put_bangla_text(image, text, position, font):
    # Convert the image to PIL format
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    # Draw the text
    draw.text(position, text, font=font, fill=(0, 140, 255))
    # Convert back to OpenCV format
    return np.array(image_pil)

# Dictionary mapping indices to the corresponding English words
index_to_word = [
    'Bad', 'Beautiful', 'Friend', 'Good', 'House',
    'Me', 'My', 'Request', 'Skin', 'Urine', 'You'
]

# Confidence threshold for displaying predictions
CONFIDENCE_THRESHOLD = 0.7

# Continuously capturing the frames from the webcam, detecting the hand landmarks, and interpreting the sign language
while True:
    # Reading a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flipping the frame horizontally
    frame = cv2.flip(frame, 1)

    # Getting the height, width, and channel count of the frame
    H, W, _ = frame.shape

    # Converting the frame from BGR color space to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(frame_rgb)

    # Resize the frame to match the model input shape
    frame_resized = cv2.resize(frame_rgb, (224, 224))

    # Normalize the frame
    frame_normalized = frame_resized / 255.0

    # Add batch dimension (model expects 4D input)
    data = np.expand_dims(frame_normalized, axis=0)

    # Predicting the sign language character using the trained model
    predictions = trainedModel.predict(data)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"Predictions: {predictions}")
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

    # Map the predicted index to the correct word
    if confidence >= CONFIDENCE_THRESHOLD:
        bengali_text = index_to_word[predicted_class]
    else:
        bengali_text = "Uncertain"

    # Draw hand landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing the landmarks with color
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Creating a white background for the text
    text_bg = np.zeros((100, W, 3), dtype=np.uint8)
    text_bg.fill(255)

    # Adding the predicted sign language character to the white background
    text_bg = put_bangla_text(text_bg, bengali_text, (W // 2.5 - 20, 10), font)

    # Concatenating the frame and the text background
    frame_with_text = np.concatenate((frame, text_bg), axis=0)

    # Displaying the frame with the detected hand landmarks and the predicted sign language character
    cv2.imshow('Sign Language Interpreter', frame_with_text)

    # Terminating the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam and closing all windows
cap.release()
cv2.destroyAllWindows()
