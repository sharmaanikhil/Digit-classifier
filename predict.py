import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("Loading model...")
try:
    m = keras.models.load_model('model/mnist_classifier.h5')
    print("Model loaded")
except:
    print("Model not found. Run train_model.py first")
    exit()

def prep_img(path):
    
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    arr = np.array(img)
    if np.mean(arr) > 127:
        arr = 255 - arr

    arr = arr.astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict(arr):
    pred = m.predict(arr, verbose=0)
    digit = np.argmax(pred)
    conf = np.max(pred) * 100
    print(f"\nPredicted: {digit}")
    print(f"Confidence: {conf:.1f}%")
    print("\nAll probabilities:")
    for i in range(10):
        p = pred[0][i] * 100
        bar = '█' * int(p / 5)
        print(f"{i}: {bar} {p:.1f}%")
    
    return digit, conf

print("\n" + "="*40)
print("Testing on MNIST data")
print("="*40)

(_, _), (x_te, y_te) = keras.datasets.mnist.load_data()

n = 5
idx = np.random.choice(len(x_te), n, replace=False)

plt.figure(figsize=(15, 3))
for i, k in enumerate(idx):
    img = x_te[k]
    real = y_te[k]
    arr = img.astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = m.predict(arr, verbose=0)
    p = np.argmax(pred)
    conf = np.max(pred) * 100
    plt.subplot(1, n, i + 1)
    plt.imshow(img, cmap='gray')
    col = 'green' if p == real else 'red'
    plt.title(f'Real: {real}\nPred: {p}\n{conf:.1f}%', color=col, fontsize=10)
    plt.axis('off')
    print(f"\nSample {i+1}: Real={real}, Pred={p}, Conf={conf:.1f}%")

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("\n\nSaved predictions.png")
plt.show()
print("\n" + "="*40)
print("Checking test_images folder")
print("="*40)

if os.path.exists('test_images'):
    files = [f for f in os.listdir('test_images') if f.endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        print(f"Found {len(files)} images")
        for f in files:
            path = os.path.join('test_images', f)
            print(f"\n{f}:")
            
            try:
                arr = prep_img(path)
                predict(arr)
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("No images found")
else:
    print("test_images folder not found")

print("\n" + "="*40)
print("Done")
print("="*40)