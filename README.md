# Handwritten Digit Recognition using MNIST

## About the Project
This project is a simple handwritten digit recognition system built using Python.  
The program is trained to recognize digits from 0 to 9 using the MNIST dataset and a basic Convolutional Neural Network (CNN).
The main idea behind this project was to understand how image data is processed and how a machine learning model can be trained to classify images.

---
## How to Run
### Step 1: Install Dependencies
Make sure Python (3.7 or above) is installed on your system.
From the project directory, run:

```bash
pip install -r requirements.txt
```

---

### Step 2: Train the Model
Run the training script:

```bash
python train_model.py
```

This script will:
- Download the MNIST dataset
- Normalize the image pixel values
- Train a CNN model for 5 epochs
- Save the trained model inside the `model/` folder

Training takes a few minutes on a normal CPU.

---

### Step 3: Test Predictions
After training, run:

```bash
python predict.py
```

The script loads the saved model and predicts digits from sample images.  
The predicted digit and confidence are displayed on the screen.

---

## Dataset Used
The MNIST dataset contains 70,000 grayscale images of handwritten digits.  
Each image is 28×28 pixels in size.  
It is a commonly used dataset for learning image classification techniques.

---
## Model Explanation

A simple CNN was used in this project.  
The convolution layers help in extracting important patterns from images, while pooling layers reduce image size.  
The final dense layers are used to classify the image into one of the 10 digit classes.
The model achieves around approx 98% accuracy on the test data.

---
## What I Learned
- How to preprocess image data for machine learning
- Why normalization is important for training stability
- How CNNs work for image classification
- How to save and reuse a trained model
- How to test the model on new images

---
## Conclusion
This project helped me understand the basics of image classification using neural networks.  
It is a simple but effective example of how machine learning can be applied to real image data.


