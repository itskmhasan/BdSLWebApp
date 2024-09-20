import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from googletrans import Translator  # Import Google Translate API
from rembg import remove
from detector.forms import SignImageForm

# Load the trained model
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'sign_language_cnn_model.keras'))

# Dictionary mapping index to sign
index_to_word = {0: 'Bad', 1: 'Beautiful', 2: 'Friend', 3: 'Good', 4: 'House', 5: 'Me', 6: 'My', 7: 'Request',
                 8: 'Skin', 9: 'Urine', 10: 'You'}

# Initialize Google Translate API
translator = Translator()


def predict_sign(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Resize to model input shape
    img_normalized = img_resized / 255.0  # Normalize
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions)  # Get the index of the highest prediction
    return index_to_word.get(predicted_class, "Unknown")


def translate_to_bengali(text):
    """Translate the detected sign to Bengali."""
    try:
        translation = translator.translate(text, src='en', dest='bn')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails


def upload_image(request):
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

            # Get the predicted sign
            predicted_sign = predict_sign(output_image_path)

            # Translate the predicted sign to Bengali
            translated_sign = translate_to_bengali(predicted_sign)

            # Pass the path of the background-removed image and the translation result to the result template
            context = {
                'form': form,
                'image_url': uploaded_image_path,
                'bg_removed_image_url': bg_removed_image_url,
                'predicted_sign': predicted_sign,  # Original sign prediction
                'translated_sign': translated_sign  # Translated sign in Bengali
            }
            return render(request, 'result.html', context)
    else:
        form = SignImageForm()

    return render(request, 'index.html', {'form': form})
