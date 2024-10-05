import pickle
import cv2
import numpy as np
import os
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from googletrans import Translator
from rembg import remove
from detector.forms import SignImageForm
import mediapipe as mp

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = model_dict['labels_dict']

translator = Translator()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)


def predict_sign(image_path):
    """Predict sign from image path and return the sign, its translation, and confidence."""
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return "Unknown", "Unknown", 0.0  # Added accuracy return

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

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

        if len(data_aux) == 42:
            data_aux.extend([0] * 42)
        elif len(data_aux) == 84:
            pass
        else:
            print(f"Unexpected data length: {len(data_aux)}")
            return "Invalid data", "Invalid data", 0.0

        prediction_probs = model.predict_proba([np.asarray(data_aux)])  # Get probabilities
        predicted_index = np.argmax(prediction_probs)  # Get the index of the highest probability
        predicted_sign = str(labels_dict[predicted_index])  # Get the predicted sign
        confidence = prediction_probs[0][predicted_index] * 100  # Get confidence as a percentage

        translated_sign = translate_to_bengali(predicted_sign)
        print(translated_sign)

        return predicted_sign, translated_sign, confidence  # Return predicted sign, translation, and confidence
    else:
        return "No hands detected", "N/A", 0.0


def translate_to_bengali(text):
    try:
        translation = translator.translate(text, src='en', dest='bn')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def upload_image(request):
    """Handle the image upload, background removal, prediction, and translation."""
    if request.method == 'POST':
        form = SignImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']
            fs = FileSystemStorage()

            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_image_path = fs.url(filename)

            input_image_path = fs.path(filename)

            with open(input_image_path, "rb") as img_file:
                input_image = img_file.read()

            output_image_data = remove(input_image)

            output_image_path = os.path.join(fs.location, 'bg_removed_' + uploaded_file.name)
            with open(output_image_path, 'wb') as output_image_file:
                output_image_file.write(output_image_data)

            predicted_sign, translated_sign, confidence = predict_sign(output_image_path)

            confidence_message = f"Confidence: {confidence:.2f}"
            accuracy = f"{confidence:.2f}%"  # Format accuracy as percentage

            context = {
                'form': form,
                'image_url': uploaded_image_path,
                'bg_removed_image_url': fs.url('bg_removed_' + uploaded_file.name),
                'predicted_sign': predicted_sign,
                'translated_sign': translated_sign,
                'confidence_message': confidence_message,
                'accuracy': accuracy,
            }
            return render(request, 'result.html', context)
    else:
        form = SignImageForm()

    return render(request, 'index.html', {'form': form})
