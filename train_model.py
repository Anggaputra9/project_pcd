import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler


def extract_features(image_file):
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
    energy_val = np.sum(hist ** 2)
    var_val = np.sum(((levels - mean_val) ** 2) * hist.flatten())
    smooth_val = 1 - 1 / (1 + var_val)

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    energy_glcm = graycoprops(glcm, 'energy')[0, 0]

    return [mean_val, skew_val, energy_val, smooth_val,
            contrast, homogeneity, ASM, energy_glcm]


def augment_data(image_file):
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Gambar {image_file} tidak ditemukan.")
    aug = [cv2.flip(img, 1)]
    for angle in [90, 180, 270]:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        aug.append(rotated)
    return aug


def main():
    image_folder = "dataset/"
    files = glob.glob(image_folder + "**/*.jpg", recursive=True)
    files += glob.glob(image_folder + "**/*.png", recursive=True)
    files += glob.glob(image_folder + "**/*.jpeg", recursive=True)

    features = []
    labels = []
    for img_file in files:
        if "matang" in img_file:
            label = 1
        elif "mentah" in img_file:
            label = 0
        else:
            continue
        features.append(extract_features(img_file))
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    if len(features) == 0:
        raise RuntimeError("Tidak menemukan gambar di folder dataset.")
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Jumlah class kurang dari 2.")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    accuracies = []
    cv_scores_list = []

    for repeat in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=repeat)

        aug_features = []
        aug_labels = []

        for idx, img_file in enumerate(files):
            if (features[idx] == X_train).all(axis=1).any():
                label = 1 if "matang" in img_file else 0
                for aug_img in augment_data(img_file):
                    temp_file = "temp.jpg"
                    cv2.imwrite(temp_file, aug_img)
                    feat = extract_features(temp_file)
                    aug_features.append(feat)
                    aug_labels.append(label)
                    os.remove(temp_file)

        if aug_features:
            aug_features = scaler.transform(aug_features)
            X_train = np.vstack([X_train, aug_features])
            y_train = np.hstack([y_train, aug_labels])

        k_vals = range(1, 11)
        scores = []
        for k in k_vals:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X_train, y_train, cv=5).mean()
            scores.append(score)

        k_best = k_vals[np.argmax(scores)]
        best_cv_score = max(scores)
        cv_scores_list.append(best_cv_score)

        knn = KNeighborsClassifier(n_neighbors=k_best)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    print(f"\n🔁 Hasil dari 10x Pengujian:")
    print(f"Rata-rata Akurasi Test Set : {np.mean(accuracies) * 100:.2f}%")
    print(f"Rata-rata CV Score         : {np.mean(cv_scores_list) * 100:.2f}%")
    print(f"Best k rata-rata            : {k_best}")

    joblib.dump({"model": knn, "scaler": scaler}, "knn_pisang_model.pkl")
    print("Model dan scaler disimpan di knn_pisang_model.pkl")


if __name__ == "__main__":
    main()
