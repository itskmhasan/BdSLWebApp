import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

DATA_DIR = './BdSL_Dataset'
print(DATA_DIR)
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            # Ensure the generated data is of a consistent length
            if len(data_aux) == 42:
                # One-hand detected, pad with zeros
                data_aux.extend([0] * 42)
                data.append(data_aux)
                labels.append(dir_)
            elif len(data_aux) == 84:
                # Two-hand detected
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Skipping inconsistent data with length: {len(data_aux)}")

# Save the data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)