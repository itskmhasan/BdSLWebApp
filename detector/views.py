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

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Dictionary mapping index to sign
labels_dict = model_dict['labels_dict']

# Initialize Google Translate API
translator = Translator()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def predict_sign(image_path):
    """Predict sign from image path and return the sign, its translation, and confidence."""
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was properly loaded
    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return "Unknown", "Unknown", 0.0  # Added accuracy return

    # Process the frame to detect hands
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Collect the normalized landmarks
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

        # Make prediction using the model
        prediction_probs = model.predict_proba([np.asarray(data_aux)])  # Get probabilities
        predicted_index = np.argmax(prediction_probs)  # Get the index of the highest probability
        predicted_sign = str(labels_dict[predicted_index])  # Get the predicted sign
        confidence = prediction_probs[0][predicted_index] * 100  # Get confidence as a percentage

        # Translate the detected sign to Bengali
        translated_sign = translate_to_bengali(predicted_sign)

        return predicted_sign, translated_sign, confidence  # Return predicted sign, translation, and confidence
    else:
        return "No hands detected", "N/A", 0.0


def translate_to_bengali(text):
    """Translate the detected sign to Bengali."""
    try:
        translation = translator.translate(text, src='en', dest='bn')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails


def upload_image(request):
    """Handle the image upload, background removal, prediction, and translation."""
    if request.method == 'POST':
        form = SignImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']
            fs = FileSystemStorage()

            # Save the original uploaded image
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_image_path = fs.url(filename)

            # Load the image to remove the background
            input_image_path = fs.path(filename)

            with open(input_image_path, "rb") as img_file:
                input_image = img_file.read()

            # Use rembg to remove the background
            output_image_data = remove(input_image)

            # Save the image with the background removed
            output_image_path = os.path.join(fs.location, 'bg_removed_' + uploaded_file.name)
            with open(output_image_path, 'wb') as output_image_file:
                output_image_file.write(output_image_data)

            # Get the predicted sign and its translation
            predicted_sign, translated_sign, confidence = predict_sign(output_image_path)

            # Prepare the response data
            confidence_message = f"Confidence: {confidence:.2f}"
            accuracy = f"{confidence:.2f}%"  # Format accuracy as percentage

            # Pass the path of the background-removed image and the prediction results to the result template
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
