import sys
import cv2
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import pygame

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Capture")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("Enter DroidCam IP and click 'Connect'")
        self.label.setGeometry(150, 10, 500, 30)
        self.label.setAlignment(Qt.AlignCenter)

        self.ip_input = QLineEdit(self)
        self.ip_input.setGeometry(10, 50, 200, 30)
        self.ip_input.setPlaceholderText('Enter IP address')

        self.connect_button = QPushButton('Connect', self)
        self.connect_button.setGeometry(220, 50, 100, 30)
        self.connect_button.clicked.connect(self.start_capture)

        self.capture_button = QPushButton('Capture', self)
        self.capture_button.setGeometry(330, 50, 100, 30)
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)

        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 100, 780, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Initialize pygame mixer for sound
        pygame.mixer.init()

    def start_capture(self):
        self.ip_input.setEnabled(False)
        self.connect_button.setEnabled(False)
        ip_address = self.ip_input.text().strip()
        if ip_address:
            self.stream_url = f'http://{ip_address}:4747/video'
            self.camera = cv2.VideoCapture(self.stream_url)

            # Create a timer to update the camera feed
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Update every 30 milliseconds

            # Timer for connection timeout
            self.connection_timer = QTimer(self)
            self.connection_timer.timeout.connect(self.check_connection)
            self.connection_timer.setSingleShot(True)
            self.connection_timer.start(5000)  # Check connection after 5 seconds

    def check_connection(self):
        if not self.camera.isOpened():
            self.timer.stop()
            self.camera.release()
            self.ip_input.setEnabled(True)
            self.connect_button.setEnabled(True)
            self.capture_button.setEnabled(False)
            QMessageBox.critical(self, "Connection Error", "Failed to connect to the camera. Please check the IP address and try again.")
    
        
    def capture_image(self):
        # Read the current frame from the camera
        ret, frame = self.camera.read()
        if ret:
            # Convert the frame to RGB format (required for QImage)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save the captured image to a file (optional)
            cv2.imwrite('captured_image.png', frame)

            # Preprocess the image for the model
              # Resize to match model's input size
            
            img = image.load_img('captured_image.png', target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
        
            
            

            # Load and compile the model (assuming 'super-model.h5' is your trained model file)
            model = tf.keras.models.load_model('./model-GoogleNet.h5')
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Make prediction
            prediction = model.predict(img_array)
            class_label = 'آفتاب گردان است' if prediction[0][0] > 0.48 else 'بنظر میاد آفتابگردان نیست'

            # Display result
            self.label.setText(f'The image is predicted as: {class_label}')
            self.label.adjustSize()
            reshaped_text = get_display(arabic_reshaper.reshape(class_label))
            print(reshaped_text)
            print(prediction)
            # Play sound if prediction is successful
            if class_label == 'آفتاب گردان است':
                pygame.mixer.music.load('ast.mp3')
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.load('nist.mp3')
                pygame.mixer.music.play()
    
    def update_frame(self):
        # Update camera feed on the label widget
        ret, frame = self.camera.read()
        if ret:
            self.connection_timer.stop()  # Stop connection timer if frame is received
            self.capture_button.setEnabled(True)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        # Clean up resources when closing the application
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'connection_timer'):
            self.connection_timer.stop()
        pygame.mixer.quit()  # Quit pygame mixer
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
