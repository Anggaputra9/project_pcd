import tkinter as tk
from tkinter import filedialog
import joblib
import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_file):
    """Ekstrak 8 fitur dari sebuah gambar."""
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Gambar {image_file} tidak ditemukan.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))  

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    levels = np.arange(256)

    mean_val = np.sum(levels * hist.flatten())  
    skew_val = skew(gray.flatten())  
    energy_val = np.sum(hist**2)  
    var_val = np.sum(((levels - mean_val) ** 2) * hist.flatten())  
    smooth_val = 1 - 1 / (1 + var_val)

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    energy_glcm = graycoprops(glcm, 'energy')[0, 0]

    return [mean_val, skew_val, energy_val, smooth_val, contrast, homogeneity, ASM, energy_glcm]

def predict():
    """Memproses gambar dan menampilkan hasil prediksi."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    img = Image.open(file_path)
    img = img.resize((250, 250))
    photo = ImageTk.PhotoImage(img)

    img_label.config(image=photo)
    img_label.image = photo
    
    features = extract_features(file_path)
    features = scaler.transform([features])  
    prediction = model.predict(features)[0]

    if prediction == 0:
        label = "Mentah"
    else:
        label = "Matang"

    result_label.config(text=f"Hasil prediksi: {label}",
                         font=('Helvetica', 18, 'bold'), fg='#ffcc5b')

# ---------------------------------------------------------------------------
# Main GUI

# Loading Model dan scaler
model_file = joblib.load('knn_pisang_model.pkl')
model = model_file["model"]
scaler = model_file["scaler"]

root = tk.Tk()
root.title("Deteksi Kematangan Pisang")
root.geometry("500x500")
root.config(bg="#edf2f4")

frame = tk.Frame(root, bg="#edf2f4")
frame.pack(pady=20)

title = tk.Label(frame, text="Deteksi Kematangan Pisang",
                 font=('Helvetica', 20, 'bold'), bg="#edf2f4")
title.pack(pady=10)

img_label = tk.Label(frame, bg="#edf2f4")
img_label.pack()

open_file = tk.Button(root, text='Pilih Gambar', command=predict,
                      font=('Helvetica', 14), bg="#ffcc5b")
open_file.pack(pady=10)

result_label = tk.Label(root, text="", font=('Helvetica', 18), bg="#edf2f4")
result_label.pack(pady=20)

root.mainloop()
