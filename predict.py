import joblib
import numpy as np
import glob
import os
import cv2
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_file):
    """Ekstrak 8 fitur dari sebuah gambar (histogram dan GLCM)."""
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Gambar {image_file} tidak ditemukan.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 150))

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

def predict_single(image_file, model, scaler):
    """Prediksi sebuah gambar saja."""
    feat = extract_features(image_file)
    feat = scaler.transform([feat]) 
    pred = model.predict(feat)[0]
    label = "Matang" if pred == 1 else "Mentah"
    print(f"{os.path.basename(image_file)} -> Prediksi: {label}")
    return label

def predict_all(folder, model, scaler):
    """Prediksi seluruh gambar di sebuah folder."""
    images = glob.glob(folder + "/**/*.jpg", recursive=True)
    images += glob.glob(folder + "/**/*.png", recursive=True)
    images += glob.glob(folder + "/**/*.jpeg", recursive=True)

    true_labels = []
    pred_labels = []

    for img_file in images:
        true = 1 if "matang" in img_file else 0
        true_labels.append(true)

        feat = extract_features(img_file)
        feat = scaler.transform([feat]) 
        pred = model.predict(feat)[0]
        pred_labels.append(pred)

        print(f"{os.path.basename(img_file)} -> Prediksi: {'Matang' if pred == 1 else 'Mentah'}")

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    if len(true_labels) > 0:
        accuracy = (true_labels == pred_labels).mean()
        print(f"Akurasi pada folder {folder}: {accuracy * 100:.2f}%")
    else:
        print("Tidak menemukan gambar di folder.")
    

def main():
    """Main untuk 1 gambar atau 1 folder."""
    model_file = joblib.load('knn_pisang_model.pkl')
    model = model_file["model"]
    scaler = model_file["scaler"]

    path = input("Masukkan path gambar atau directory (folder), atau ketik 'all' untuk folder test: ")

    if path == "all":
        predict_all("dataset/test", model, scaler)
    elif os.path.isdir(path):
        predict_all(path, model, scaler)
    elif os.path.exists(path):
        predict_single(path, model, scaler)
    else:
        print("Path tidak ditemukan.")
    

if __name__ == '__main__':
    main()
