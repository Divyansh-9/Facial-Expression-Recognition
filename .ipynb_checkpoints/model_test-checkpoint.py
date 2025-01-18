import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Face Recognition App")
        
        # Initialize variables
        self.video = cv2.VideoCapture(0)
        self.running = False
        
        # Initialize face detection and recognition components
        self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("Trainer.yml")
        self.name_list = ["", "Divyansh"]
        
        # Initialize emotion recognition
        self.emotion_model = tf.keras.models.load_model('emotion_model.h5')
        self.emotion_dict = {0:'Angry', 1:'Disgusted', 2:'Fearful', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprised'}
        
        # Load the static image
        try:
            self.static_image = Image.open("solo.jpg")
            self.static_image = self.static_image.resize((600, 400), Image.Resampling.LANCZOS)
            self.static_image_tk = ImageTk.PhotoImage(self.static_image)
        except Exception as e:
            print(f"Error loading static image: {e}")
            self.static_image_tk = None
        
        # UI Elements
        self.video_label = Label(self.root)
        self.video_label.pack()

        self.start_button = Button(self.root, text="Start Camera", command=self.start_camera)
        self.start_button.pack()
        
        self.stop_button = Button(self.root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack()
        
        self.exit_button = Button(self.root, text="Exit", command=self.root.quit)
        self.exit_button.pack()
        
        if self.static_image_tk:
            self.display_image_with_border(self.static_image)

    def start_camera(self):
        self.running = True
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.static_image_tk:
            self.display_image_with_border(self.static_image)
        else:
            print("No static image to display.")

    def update_frame(self):
        if self.running:
            ret, frame = self.video.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.facedetect.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    try:
                        # Face Recognition
                        serial, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                        
                        # Emotion Recognition
                        roi_gray = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                        roi_gray = roi_gray.astype('float')/255.0
                        roi_gray = np.expand_dims(np.expand_dims(roi_gray, axis=0), axis=-1)
                        emotion = self.emotion_dict[np.argmax(self.emotion_model.predict(roi_gray))]
                        
                        if conf > 50:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{self.name_list[serial]} - {emotion}", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, f"Unknown - {emotion}", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Error in recognition: {e}")
                        continue

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            self.root.after(10, self.update_frame)

    def display_image_with_border(self, image):
        border_color = (255, 0, 0) 
        border_thickness = 20
        
        new_width = image.width + 2 * border_thickness
        new_height = image.height + 2 * border_thickness
        new_image = Image.new('RGB', (new_width, new_height), border_color)
        new_image.paste(image, (border_thickness, border_thickness))
        
        self.static_image_tk = ImageTk.PhotoImage(new_image)
        self.video_label.config(image=self.static_image_tk)
        self.video_label.image = self.static_image_tk

    def __del__(self):
        self.video.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
