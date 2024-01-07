from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.lang import Builder
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, save_img
import numpy as np

# Set up Kivy's window size
from kivy.config import Config
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '600')

# Load Kivy language file (optional, for UI layout)
Builder.load_string('''
<CameraApp>:
    orientation: 'vertical'
    
    Camera:
        id: camera
        index: 0  # Set the camera index explicitly
        resolution: (48, 48)
        play: True
    
    BoxLayout:
        orientation: 'horizontal'
        
        Button:
            text: 'Capture'
            on_press: root.capture()
            
        Button:
            text: 'Analyze Emotion'
            on_press: root.analyze_emotion()
            
        Image:
            id: img_preview
            source: ''
    
    Label:
        id: label_result
        text: 'Emotion: '
''')


class CameraApp(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraApp, self).__init__(**kwargs)
        self.model = load_model("facialmodel.h5")

    def capture(self):
        camera = self.ids.camera
        img_preview = self.ids.img_preview

        # Capture the image and display it in the Image widget
        print("Capturing image...")
        camera.export_to_png("captured_image.png")
        img_preview.source = "captured_image.png"
        img_preview.reload()
        print("Image captured successfully.")

    def analyze_emotion(self):
        captured_image_path = "captured_image.jpg"

        # Load the image and resize it to (48, 48)
        img = load_img(captured_image_path,
                       color_mode="grayscale", target_size=(48, 48))

        # Convert the resized image to array
        img_array = img_to_array(img)

        # Add an extra channel to make it rank 4
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # Normalize the image data
        preprocessed_image = img_array / 255.0

        # Make predictions using the loaded model
        predictions = self.model.predict(preprocessed_image)

        # Process predictions and update UI (e.g., update label text)
        emotion_labels = ['angry', 'disgust', 'fear',
                          'happy', 'neutral', 'sad', 'surprise']
        predicted_emotion_index = np.argmax(predictions)
        predicted_emotion = emotion_labels[predicted_emotion_index]

        # Update UI (e.g., set label text to the predicted emotion)
        self.ids.label_result.text = f"Emotion: {predicted_emotion}"


class EmotionDetectionApp(App):
    def build(self):
        return CameraApp()


if __name__ == '__main__':
    EmotionDetectionApp().run()
