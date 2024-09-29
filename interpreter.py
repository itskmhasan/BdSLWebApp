import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained Faster R-CNN model
model = tf.saved_model.load('path/to/your/saved_model_directory')

# Starting the webcam
cap = cv2.VideoCapture(0)

# Initialize the hand landmarks detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.7

# Continuously capturing frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Process the frame for hand detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare the frame for prediction
    input_tensor = tf.convert_to_tensor([frame])
    detections = model(input_tensor)

    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    # Draw bounding boxes for detected signs
    for i in range(len(scores)):
        if scores[i] > CONFIDENCE_THRESHOLD:
            box = boxes[i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            (ymin, xmin, ymax, xmax) = box.astype("int")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Sign Language Interpreter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
