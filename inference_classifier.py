import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

labels_dict = model_dict['labels_dict']

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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

        if len(data_aux) == 42:
            data_aux.extend([0] * 42)
        elif len(data_aux) == 84:
            pass
        else:
            print(f"Unexpected data length: {len(data_aux)}")
            continue

        try:
            prediction_probs = model.predict_proba([np.asarray(data_aux)])  # Get probabilities
            predicted_index = np.argmax(prediction_probs)  # Get the index of the highest probability
            predicted_character = str(labels_dict[predicted_index])  # Get the predicted character
            confidence = prediction_probs[0][predicted_index] * 100  # Get confidence as a percentage

            x1 = int(min(x_) * W) - 5
            y1 = int(min(y_) * H) - 5
            x2 = int(max(x_) * W) - 20
            y2 = int(max(y_) * H) - 20

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
