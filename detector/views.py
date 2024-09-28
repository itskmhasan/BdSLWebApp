import cv2
import numpy as np
import tensorflow as tf
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from googletrans import Translator
from rembg import remove
from detector.forms import SignImageForm
import mediapipe as mp
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'sign_language_cnn_model.keras'))

# Dictionary mapping index to sign
# Dictionary mapping index to sign
index_to_word = {
    0: 'Anger', 1: 'Fear', 2: 'Grateful', 3: 'Hatred', 4: 'Hope',
    5: 'Joy', 6: 'Love', 7: 'Sadness', 8: 'Shame', 9: 'Trust'
}

# Initialize Google Translate API
translator = Translator()


def predict_sign(image_path):
    """Predict sign from image path and return the sign, its translation, and accuracy."""
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was properly loaded
    if img is None:
        print(f"Error: Could not read the image from {image_path}")
        return "Unknown", "Unknown", 0.0  # Added accuracy return

    # Preprocess the image for the model
    img_resized = cv2.resize(img, (224, 224))  # Resize to match the model's input shape
    img_normalized = img_resized / 255.0  # Normalize the image
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions)  # Get the index of the highest prediction
    predicted_sign = index_to_word.get(predicted_class, "Unknown")
    confidence = np.max(predictions)  # Get the confidence of the prediction

    # Translate the detected sign to Bengali
    translated_sign = translate_to_bengali(predicted_sign)

    return predicted_sign, translated_sign, confidence  # Return confidence as well



def translate_to_bengali(text):
    """Translate the detected sign to Bengali."""
    try:
        # Translate the sign to Bengali using Google Translate
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

            # Generate the URL for the image with the background removed
            bg_removed_image_url = fs.url('bg_removed_' + uploaded_file.name)

            # Get the predicted sign and its translation
            predicted_sign, translated_sign, confidence = predict_sign(output_image_path)

            # Handle no hand detection message
            if predicted_sign == "No hands detected":
                confidence_message = "No hands detected"
                accuracy = "N/A"  # Set accuracy as not available
            else:
                confidence_message = f"Confidence: {confidence:.2f}"
                accuracy = f"{confidence:.2%}"  # Format accuracy as percentage

            # Pass the path of the background-removed image and the prediction results to the result template
            context = {
                'form': form,
                'image_url': uploaded_image_path,
                'bg_removed_image_url': bg_removed_image_url,
                'predicted_sign': predicted_sign,
                'translated_sign': translated_sign,
                'confidence_message': confidence_message,
                'accuracy': accuracy,
            }
            return render(request, 'result.html', context)
    else:
        form = SignImageForm()

    return render(request, 'index.html', {'form': form})
