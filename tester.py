import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load the pickled data dictionary containing the normalized landmark data and corresponding labels
data_dict = pickle.load(open('./signdata.pickle', 'rb'))

# convert the data and labels lists from the data dictionary to numpy arrays
data = np.array(data_dict['data'], dtype=object)
labels = np.asarray(data_dict['labels'])

# pad the sequences in the data array with zeros so that all samples have the same number of features
max_len = max(len(sample) for sample in data)
data_padded = np.zeros((len(data), max_len))
for i, sample in enumerate(data):
    data_padded[i, :len(sample)] = sample

# split the data and labels into training and testing sets with a test size of 20%, shuffling the data and using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# create a new random forest classifier model
trainedModel = RandomForestClassifier()

# train the model using the training data and labels
trainedModel.fit(x_train, y_train)

# use the trained model to predict the labels of the test data
y_predict = trainedModel.predict(x_test)

# calculate the accuracy of the model's predictions on the test data
score = accuracy_score(y_predict, y_test)

# print the accuracy score as a percentage
print('{}% of samples were classified accurately.'.format(score * 100))

# save the trained model as a pickled file
f = open('trainedModel.p', 'wb')
pickle.dump({'trainedModel': trainedModel}, f)
f.close()


#----------old_interpreter.py-----------#

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import pickle
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model_dict = pickle.load(open('./trainedModel.p', 'rb'))
trainedModel = model_dict['trainedModel']

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


# Dictionary mapping English words to Bengali words
english_to_bengali = {
    '1': 'এক',
    '2': 'দুই',
    '3': 'তিন',
    '4': 'চার',
    '5': 'পাঁচ',
    'OKAY': 'ঠিক আছে',
    # Add more mappings as needed
}

# Continuously capturing the frames from the webcam, detecting the hand landmarks and interpreting the sign language
while True:

    # Reading a frame from the webcam
    ret, frame = cap.read()
    # Flipping the frame horizontally
    frame = cv2.flip(frame, 1)

    # Getting the height, width and channel count of the frame
    H, W, _ = frame.shape

    # Converting the frame from BGR color space to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting the hand landmarks in the frame
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        all_landmarks = []

        # Draw the hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extracting the hand landmarks' positions and normalizing them
            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
            all_landmarks.extend(landmarks)

        # Flatten the list of landmarks and normalize
        data = np.array(all_landmarks).flatten()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # If we have fewer than 84 features, pad with zeros
        if len(data) < 84:
            data = np.pad(data, (0, 84 - len(data)), 'constant')

        # If we have more than 84 features, truncate to 84
        if len(data) > 84:
            data = data[:84]

        # Predicting the sign language character using the trained model
        predicted_character = trainedModel.predict([data])[0]

        bengali_text = english_to_bengali.get(predicted_character, predicted_character)

        # Creating a white background for the text
        text_bg = np.zeros((100, W, 3), dtype=np.uint8)
        text_bg.fill(255)

        # Adding the predicted sign language character to the white background
        text_bg = put_bangla_text(text_bg, bengali_text, (W // 2.5 - 20, 10), font)


    # If no hand landmarks are detected
    else:
        # Creating an empty white background for the text
        text_bg = np.zeros((100, W, 3), dtype=np.uint8)
        text_bg.fill(255)

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