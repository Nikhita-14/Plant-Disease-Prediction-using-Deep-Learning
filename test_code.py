import tkinter as tk
from tkinter import filedialog, messagebox
from keras.models import model_from_json
import numpy as np
import cv2
import os

# === 1) Load CNN model ===
model_json_path = 'model/model.json'
model_weights_path = 'model/model.weights.h5'  # Updated to match new format

# Ensure model files exist before loading
if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Model Not Found", "Model files not found.\nPlease run 'train.py' first.")
    exit()

# Load architecture
with open(model_json_path, "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)

# Load weights
classifier.load_weights(model_weights_path)

# Define your class labels (adjust if you have more)
class_labels = ['Disease', 'Healthy']

# === 2) Prediction function ===
def predict_image():
    # Open file dialog
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Load and preprocess image
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not read the selected image.")
        return

    img = cv2.resize(img, (32, 32))
    img = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255.0

    # Predict
    preds = classifier.predict(img)
    pred_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    result = f"Predicted Class: {class_labels[pred_index]}\nConfidence: {confidence:.2f}%"
    messagebox.showinfo("Prediction Result", result)

# === 3) Build GUI ===
root = tk.Tk()
root.title("CNN Image Classifier")

label = tk.Label(root, text="Load a query image to classify", font=("Arial", 14))
label.pack(pady=20)

predict_button = tk.Button(root, text="Select Image and Predict", command=predict_image, font=("Arial", 12))
predict_button.pack(pady=10)

root.mainloop()