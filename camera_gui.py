import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from speak import speakTrash

MODEL_PATH = './models/waste_classification_model_20250619_215017.h5'
model = tf.keras.models.load_model(MODEL_PATH)
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

def predict_frame(frame):
    processed = preprocess_frame(frame)
    preds = model.predict(processed)
    class_idx = np.argmax(preds[0])
    return CATEGORIES[class_idx]

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("약시인들을 위한 분리수거 쓰레기 분류기")
        self.window.configure(bg="white")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Rounded.TButton",
                        font=("Arial Rounded MT Bold", 16, "bold"),  # 글자 굵게 변경
                        padding=12,
                        relief="flat",
                        borderwidth=0,
                        background="#006400",
                        foreground="white")
        style.map("Rounded.TButton",
                  background=[('active', '#228B22')])

        self.cap = cv2.VideoCapture(0)

        self.label = tk.Label(window, bg="white")
        self.label.pack(pady=(10, 5))

        self.btn = ttk.Button(window, text="분류하기", command=self.classify, style="Rounded.TButton")
        self.btn.pack(pady=(20, 10))

        self.result_text = tk.Label(window, text="", font=("Arial Rounded MT Bold", 20), fg="#006400", bg="white")
        self.result_text.pack(pady=(5, 10))

        self.current_frame = None
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    def classify(self):
        if self.current_frame is not None:
            result = predict_frame(self.current_frame)
            print(f"✅ 분류 결과: {result}")
            self.result_text.config(text=f"분류 결과: {result}")
            speakTrash(result)
            self.window.after(5000, lambda: self.result_text.config(text=""))

    def on_closing(self):
        self.cap.release()
        self.window.destroy()

root = tk.Tk()
app = App(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()