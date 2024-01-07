import os
import random
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Load the model with the optimizer state
model = load_model("emotion_detection_model.h5")

# Path to the folder containing test images
test_folder_path = "fer2013/test"

# Get a list of all subfolders in the test folder
emotion_categories = os.listdir(test_folder_path)

# Randomly select an emotion category
random_emotion_category = random.choice(emotion_categories)

# Construct the full path to the randomly selected emotion category folder
emotion_category_folder = os.path.join(
    test_folder_path, random_emotion_category)

# Get a list of all image files in the selected emotion category folder
image_files = [f for f in os.listdir(
    emotion_category_folder) if f.endswith(".jpg")]

# Randomly select an image file from the emotion category folder
random_image_file = random.choice(image_files)

# Construct the full path to the randomly selected image
captured_image_path = os.path.join(emotion_category_folder, random_image_file)

# Load the image in grayscale
img = load_img(captured_image_path, color_mode="grayscale",
               target_size=(48, 48))

# Convert the grayscale image to array
img_array = img_to_array(img)

# Add an extra dimension to make it a rank 4 array
img_array = np.expand_dims(img_array, axis=0)

# Continue with emotion analysis...
preprocessed_image = img_array / 255.0  # Normalize the image data

# Make predictions using the loaded model
predictions = model.predict(preprocessed_image)

# Process predictions and update UI (e.g., update label text)
emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']
predicted_emotion_index = np.argmax(predictions)
predicted_emotion = emotion_labels[predicted_emotion_index]

# Update UI (e.g., set label text to the predicted emotion)
print(
    f"Predicted Emotion for {random_emotion_category}/{random_image_file}: {predicted_emotion}")
