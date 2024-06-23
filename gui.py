import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

def AnimalEmotionModel (Model.json, model.h5):
    with open(Model.json, "r") as file:
        loaded_model_json = file.read()
        model = tf.keras.models.model_from_json(loaded_model_json)

    model.load_weights(model.h5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def Detect(file_path):
    global label1, sign_image

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            face_roi = gray_image[y:y+h, x:x+w]
            resized_roi = cv2.resize(face_roi, (48, 48))
            resized_roi = np.expand_dims(resized_roi, axis=-1)
            resized_roi = np.expand_dims(resized_roi, axis=0) / 255.0  # Normalize
            pred_index = np.argmax(model.predict(resized_roi))
            emotion = Emotions_list[pred_index]
            label1.config(text="Predicted Emotion: " + emotion)
    except Exception as e:
        label1.config(text='Unable to detect')

def show_Detect_button(file_path):
    detect_button = tk.Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=10)
    detect_button.configure(background="white", foreground="black", font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        uploaded_image.thumbnail(((top.winfo_width() / 2.3), (top.winfo_height() / 2.3)))
        photo = ImageTk.PhotoImage(uploaded_image)

        sign_image.config(image=photo)
        sign_image.image = photo
        label1.config(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)

# GUI setup
top = tk.Tk()
top.geometry('800x800')
top.title("Animal Emotion Detection")
top.configure(background='white')

label1 = tk.Label(top, background="white", font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

Emotions_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Load the model
model = AnimalEmotionModel("Model.json", "model.h5")

upload_button = tk.Button(top, text="Upload Image", command=upload_image, padx=10, pady=10)
upload_button.configure(background="white", foreground="black", font=('arial', 10, 'bold'))
upload_button.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = tk.Label(top, text="Animal Emotion Detection", pady=20, font=('arial', 16, 'bold'))
heading.configure(background='white', foreground='black')
heading.pack()

top.mainloop()
