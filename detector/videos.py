import pickle
import cv2
import numpy as np
import mediapipe as mp
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
import os
from googletrans import Translator

model_dict = pickle.load(open(os.path.join(settings.BASE_DIR, 'model.p'), 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = model_dict['labels_dict']

translator = Translator()

latest_translated_text = ""


def video_predict_sign(data_aux):
    prediction_probs = model.predict_proba([np.asarray(data_aux)])  # Get probabilities
    predicted_index = np.argmax(prediction_probs)  # Get the index of the highest probability
    predicted_character = str(labels_dict[predicted_index])  # Get the predicted character
    confidence = prediction_probs[0][predicted_index] * 100  # Get confidence as a percentage
    return predicted_character, confidence


def translate_text(text, src_lang='en', dest_lang='bn'):
    """Translate text from the source language to the destination language (e.g., Bengali)."""
    try:
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def video_stream():
    global latest_translated_text
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

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

            x1 = int(min(x_) * W) - 5
            y1 = int(min(y_) * H) - 5
            x2 = int(max(x_) * W) - 20
            y2 = int(max(y_) * H) - 20

            try:
                predicted_character, confidence = video_predict_sign(data_aux)

                latest_translated_text = translate_text(predicted_character, dest_lang='bn')

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255),
                #             2, cv2.LINE_AA)
                cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error during prediction: {e}")

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Set JPEG quality to 50 for faster encoding
        ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()


def get_translation(request):
    """API endpoint to send the latest translated text."""
    global latest_translated_text
    return JsonResponse({'translated_text': latest_translated_text})


def video_feed(request):
    return StreamingHttpResponse(video_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    return render(request, 'videos.html')
