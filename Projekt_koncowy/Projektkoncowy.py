import numpy as np
print(np.__version__)

# mnist_mlp_full.ipynb
# Rozpoznawanie cyfr MNIST z użyciem MLPClassifier i Keras (dropout)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# 1. Ładowanie danych
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0  # Normalizacja
y = y.astype(np.uint8)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sprawdzenie wpływu różnych random_state na uczenie
seeds = [0, 42]
init_results = {}

for seed in seeds:
    print(f"Trening z random_state={seed}")
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20, random_state=seed, verbose=False)
    clf.fit(X_train[:5000], y_train[:5000])  # mały zbiór do szybkiego testu

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, clf.predict_proba(X_test))
    init_results[seed] = (acc, loss)

    print(f"Seed={seed} | Accuracy={acc:.4f}, LogLoss={loss:.4f}")
   
    # Analiza wpływu rozmiaru zbioru treningowego na Accuracy i LogLoss (test i walidacja)
train_sizes = [1000, 5000, 10000, 30000]
size_results = {}

for n in train_sizes:
    print(f"\nTrening na {n} próbkach")

    # Ręczny podział na trening i walidację (90% / 10%)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train[:n], y_train[:n], test_size=0.1, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20, random_state=42)
    clf.fit(X_tr, y_tr)

    acc_test = accuracy_score(y_test, clf.predict(X_test))
    loss_test = log_loss(y_test, clf.predict_proba(X_test))

    acc_val = accuracy_score(y_val, clf.predict(X_val))
    loss_val = log_loss(y_val, clf.predict_proba(X_val))

    size_results[n] = {
        'test_acc': acc_test, 'test_loss': loss_test,
        'val_acc': acc_val, 'val_loss': loss_val
    }

    print(f"Test Accuracy={acc_test:.4f}, Val Accuracy={acc_val:.4f}")
    print(f"Test Loss={loss_test:.4f}, Val Loss={loss_val:.4f}")
# Wykresy: Accuracy i LogLoss vs. rozmiar zbioru treningowego
plt.figure()
plt.plot(train_sizes, [size_results[n]['test_acc'] for n in train_sizes], marker='o', label='Test Accuracy')
plt.plot(train_sizes, [size_results[n]['val_acc'] for n in train_sizes], marker='s', label='Val Accuracy')
plt.title("Accuracy vs. Training Size")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(train_sizes, [size_results[n]['test_loss'] for n in train_sizes], marker='o', label='Test Loss')
plt.plot(train_sizes, [size_results[n]['val_loss'] for n in train_sizes], marker='s', label='Val Loss')
plt.title("LogLoss vs. Training Size")
plt.xlabel("Training size")
plt.ylabel("LogLoss")
plt.legend()
plt.grid(True)
plt.show()

# Early stopping z podziałem wewnętrznym na walidację
clf = MLPClassifier(hidden_layer_sizes=(512, 256), early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=5, random_state=42)
clf.fit(X_train, y_train)

print(f"Liczba epok: {clf.n_iter_}")

# Wykresy: Training loss + Val accuracy
plt.figure()
plt.plot(clf.loss_curve_)
plt.title("Training Loss (EarlyStopping)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(clf.validation_scores_)
plt.title("Validation Accuracy (EarlyStopping)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Budowa modelu z dropoutem
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Kompilacja
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Trenowanie
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50,
                    batch_size=128, callbacks=[early_stop], verbose=2)

# Wykresy: accuracy i loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss with Dropout + EarlyStopping')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy with Dropout + EarlyStopping')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
