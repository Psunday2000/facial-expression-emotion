from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.lang import Builder

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
        resolution: (640, 480)
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
        id: emotion_label
        text: 'Emotion: '
''')


class CameraApp(BoxLayout):
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
        # Dummy function for emotion analysis
        emotion_result = "Happy"  # Replace with your actual emotion analysis logic
        self.ids.emotion_label.text = f'Emotion: {emotion_result}'


class EmotionDetectionApp(App):
    def build(self):
        return CameraApp()


if __name__ == '__main__':
    EmotionDetectionApp().run()
