import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('model'):
    os.makedirs('model')

print("Loading data...")
(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()

print(f"Got {x_tr.shape[0]} training images")

x_tr = x_tr.astype('float32') / 255.0
x_te = x_te.astype('float32') / 255.0

x_tr = x_tr.reshape(-1, 28, 28, 1)
x_te = x_te.reshape(-1, 28, 28, 1)

y_tr = keras.utils.to_categorical(y_tr, 10)
y_te = keras.utils.to_categorical(y_te, 10)

print("\nBuilding model...")
m = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

m.summary()

print("\nTraining...")
h = m.fit(x_tr, y_tr, batch_size=128, epochs=5, validation_split=0.1, verbose=1)

print("\nChecking accuracy...")
loss, acc = m.evaluate(x_te, y_te, verbose=0)
print(f"Accuracy: {acc * 100:.2f}%")

m.save('model/mnist_classifier.h5')
print("\nSaved model")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training plot")

print("\nTesting some samples...")
idx = np.random.choice(len(x_te), 5, replace=False)

for i in idx:
    img = x_te[i]
    real = np.argmax(y_te[i])
    pred = m.predict(img.reshape(1, 28, 28, 1), verbose=0)
    p = np.argmax(pred)
    conf = np.max(pred) * 100
    print(f"Real: {real}, Predicted: {p}, Confidence: {conf:.1f}%")

print("\nDone!")