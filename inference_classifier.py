import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize video capture (use correct backend)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Allow detection of up to 2 hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

# Labels dictionary (mapping prediction index to sign)
labels_dict = model_dict['labels_dict']

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get the frame's dimensions
    H, W, _ = frame.shape

    # Convert the image from BGR (OpenCV default) to RGB (MediaPipe uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Process each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect the normalized landmarks for the hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Check if one hand (42 values) or two hands (84 values) were detected
        if len(data_aux) == 42:
            # Pad with zeros to make it 84 features
            data_aux.extend([0] * 42)
        elif len(data_aux) == 84:
            # Two-hand gesture, no need to modify data_aux
            pass
        else:
            # Handle unexpected cases
            print(f"Unexpected data length: {len(data_aux)}")
            continue

        try:
            # Make prediction using the model
            prediction_probs = model.predict_proba([np.asarray(data_aux)])  # Get probabilities
            predicted_index = np.argmax(prediction_probs)  # Get the index of the highest probability
            predicted_character = str(labels_dict[predicted_index])  # Get the predicted character
            confidence = prediction_probs[0][predicted_index] * 100  # Get confidence as a percentage

            # Calculate bounding box for drawing
            x1 = int(min(x_) * W) - 5
            y1 = int(min(y_) * H) - 5
            x2 = int(max(x_) * W) - 20
            y2 = int(max(y_) * H) - 20

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")

    # Display the frame with drawings
    cv2.imshow('frame', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
