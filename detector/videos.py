import cv2
import numpy as np
import tensorflow as tf
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.conf import settings
import os

# Assuming you are using MediaPipe for hand tracking
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'sign_language_cnn_model.keras'))

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
index_to_word = {
    0: 'Anger', 1: 'Fear', 2: 'Grateful', 3: 'Hatred', 4: 'Hope',
    5: 'Joy', 6: 'Love', 7: 'Sadness', 8: 'Shame', 9: 'Trust'
}


def video_predict_sign(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Prepare the image for prediction
        img_resized = cv2.resize(frame, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Make predictions
        predictions = model.predict(img_batch)
        predicted_class = np.argmax(predictions)
        predicted_sign = index_to_word.get(predicted_class, "Unknown")
        confidence = np.max(predictions)  # Get the confidence score

        return predicted_sign, confidence
    else:
        return "No hands detected", 0.0  # Return 0 confidence if no hands are detected


def video_stream():
    cap = cv2.VideoCapture(0)  # Capture from the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and bounding box if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the bounding box coordinates
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Draw bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Predict the sign and get confidence
        predicted_sign, confidence = video_predict_sign(frame)

        # Overlay the predicted sign and confidence on the frame
        cv2.putText(frame, f"Sign: {predicted_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.4f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # Encode the frame and send it to the web page
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()


# View for real-time video feed
def video_feed(request):
    return StreamingHttpResponse(video_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# Home page view
def home(request):
    return render(request, 'videos.html')
