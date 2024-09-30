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
from PIL import ImageFont, ImageDraw, Image

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

# Load the SolaimanLipi font (ensure the path is correct)
font_path = "SolaimanLipi_22-02-2012.ttf"  # Ensure the font file is available in your project
font = ImageFont.truetype(font_path, 32)  # Adjust the size as needed


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

        translated_sign = translate_to_bengali(predicted_sign)

        return predicted_sign, translated_sign, confidence  # Return predicted sign, translation, and confidence
    else:
        return "No hands detected", "N/A", 0.0


def translate_to_bengali(text):
    """Translate the predicted sign to Bengali using Google Translate."""
    try:
        translation = translator.translate(text, src='en', dest='bn')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def put_bengali_text_on_image(image_path, translated_text, position=(50, 100)):
    """Overlay Bengali text on the image using the SolaimanLipi font."""
    # Open the image
    img = cv2.imread(image_path)

    # Convert the OpenCV image to a PIL image for text rendering
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Create a drawing context
    draw = ImageDraw.Draw(pil_img)

    # Draw the Bengali text using SolaimanLipi font
    draw.text(position, translated_text, font=font, fill=(255, 255, 255))  # White text

    # Convert the image back to OpenCV format
    img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Save or display the image with the text (you can choose to save it if necessary)
    output_path = image_path.replace('.jpg', '_with_text.jpg')  # Modify for your file extension
    cv2.imwrite(output_path, img_with_text)

    return output_path  # Return the path of the image with the text


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

            # Generate the image with Bengali text overlay
            final_image_with_text_path = put_bengali_text_on_image(output_image_path, translated_sign)

            # Prepare the response data
            confidence_message = f"Confidence: {confidence:.2f}%"
            accuracy = f"{confidence:.2f}%"  # Format accuracy as percentage

            # Pass the path of the background-removed image and the prediction results to the result template
            context = {
                'form': form,
                'image_url': uploaded_image_path,
                'bg_removed_image_url': fs.url('bg_removed_' + uploaded_file.name),
                'final_image_with_text_url': fs.url(final_image_with_text_path),  # Add the final image with text
                'predicted_sign': predicted_sign,
                'translated_sign': translated_sign,
                'confidence_message': confidence_message,
                'accuracy': accuracy,
            }
            return render(request, 'result.html', context)
    else:
        form = SignImageForm()

    return render(request, 'index.html', {'form': form})
