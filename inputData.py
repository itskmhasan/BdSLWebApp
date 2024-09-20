import cv2
import os

# Different letters of sign language
class_names = ["OKAY", "1", "2"]

# Number of samples to capture for each letter
num_samples_per_class = 300

# Image dimensions for CNNs
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Create a new directory to store the images
dataset_dir = "sign_language_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Set font and text properties for prompt text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (153, 204, 255)  # Baby blue
lineType = 2
text_background_color = (255, 255, 255)  # White

# Loop over each class (letter) and capture num_samples_per_class images for each class
for class_name in class_names:
    # Create a directory for this letter if it doesn't already exist
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Wait for the user to press 'r' to start recording
    prompt_text = f"Press 'r' to start recording {class_name}..."
    while True:
        ret, frame = camera.read()
        # Get the text size and calculate the bottom center position of the camera window
        (text_width, text_height), _ = cv2.getTextSize(prompt_text, font, fontScale, lineType)
        bottomLeftCornerOfText = ((frame.shape[1] - text_width) // 2, frame.shape[0] - 50)

        # Draw a white filled rectangle behind the prompt text
        cv2.rectangle(frame, (bottomLeftCornerOfText[0] - 5, bottomLeftCornerOfText[1] - text_height - 5),
                      (bottomLeftCornerOfText[0] + text_width + 5, bottomLeftCornerOfText[1] + 5),
                      text_background_color, -1)

        # Display the prompt text at the bottom center of the camera window
        cv2.putText(frame, prompt_text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow("Sign Language Dataset Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            break

    # Capture and save num_samples_per_class frames for this class (letter)
    for i in range(num_samples_per_class):
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize the image to the target size
        img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

        # Construct the image path and name (e.g., "OKAY_0.jpg", "OKAY_1.jpg", etc.)
        image_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")

        # Save the image to disk
        cv2.imwrite(image_path, img_resized)

        # Display the recording in the same window
        cv2.imshow("Sign Language Dataset Collection", img_resized)
        cv2.waitKey(1)

# Clean up
camera.release()
cv2.destroyAllWindows()