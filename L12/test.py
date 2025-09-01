import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

# 1. Wczytanie danych MNIST (30 000 cyfr)
print("Pobieranie danych MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_full, y_full = mnist["data"], mnist["target"].astype(int)
X, y = X_full[:30000], y_full[:30000]

# 2. Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Trening SVC z szybkim kernelem
print("Trenowanie modelu SVC na 30 000 próbkach...")
model = SVC(kernel="linear", C=5)
model.fit(X_train, y_train)

# 5. Funkcja do tworzenia 4-cyfrowych tablic
def generate_license_plates(X_raw, y_raw, count=5):
    plates = []
    labels = []
    for _ in range(count):
        idxs = random.choices(range(len(X_raw)), k=4)
        digits = [X_raw[i].reshape(28, 28) for i in idxs]
        plate = np.hstack(digits)
        label = ''.join(str(y_raw[i]) for i in idxs)
        plates.append(plate)
        labels.append(label)
    return plates, labels

# 6. Generowanie tablic
num_plates = 5000
print(f"Generowanie {num_plates} tablic rejestracyjnych...")
plates, true_labels = generate_license_plates(X, y, count=num_plates)

# 7. Przewidywanie cyfr z tablic
predicted_labels = []
for plate in plates:
    digits = np.hsplit(plate, 4)
    pred = ""
    for d in digits:
        d_flat = d.reshape(1, -1)
        d_scaled = scaler.transform(d_flat)
        p = model.predict(d_scaled)[0]
        pred += str(p)
    predicted_labels.append(pred)

# 8. Wyświetlanie 4 losowych tablic
print("Wyświetlanie 4 losowych tablic (spośród 1000):")
sample_indices = random.sample(range(num_plates), 4)
for i in sample_indices:
    img, true, pred = plates[i], true_labels[i], predicted_labels[i]
    plt.imshow(img, cmap="gray")
    plt.title(f"Prawidłowo: {true} | Przewidziano: {pred}")
    plt.axis("off")
    plt.show()

# 9. Obliczanie skuteczności
total_digits = len(true_labels) * 4
correct_digits = sum(
    int(p == t)
    for pred, true in zip(predicted_labels, true_labels)
    for p, t in zip(pred, true)
)
correct_plates = sum(p == t for p, t in zip(predicted_labels, true_labels))

plate_accuracy = correct_plates / len(true_labels)
digit_accuracy = correct_digits / total_digits

print(f"\nAnaliza na podstawie {num_plates} tablic (łącznie {total_digits} cyfr):")
print(f"Digit Accuracy: {digit_accuracy:.2%}")
print(f"Plate Accuracy: {plate_accuracy:.2%}")
