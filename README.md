# Arabic Handwritten Digit Recognition (CNN)

## 📌 Description
This project implements a computer vision system for recognizing Arabic handwritten digits (0–9) using a Convolutional Neural Network (CNN).

## 🎯 Objective
To build an accurate and efficient digit classification model by combining image preprocessing with deep learning techniques.

## ⚙️ Methodology

### 1. Data Preprocessing
- Converted images to grayscale
- Applied Otsu thresholding for digit extraction
- Cropped and centered digits
- Resized to 64×64 images

### 2. Model Architecture
- 3 Convolutional blocks (32, 64, 128 filters)
- ReLU activation + Batch Normalization
- Max pooling layers
- Fully connected layer (128 neurons)
- Dropout (0.5) to reduce overfitting
- Softmax output layer

### 3. Training Setup
- Optimizer: Adam (learning rate = 0.001)
- Epochs: up to 25 (early stopping applied)
- Dataset split: 70% training, 15% validation, 15% testing

## 📊 Results
- Test Accuracy: **94.73%**
- Strong precision, recall, and F1-score across classes

## 🧠 Key Insights
- Preprocessing significantly improved performance
- Compact CNN was sufficient (no need for large models)
- Most confusion occurred between similar digits (e.g., 2 and 3)

## 🛠️ Technologies
- Python
- NumPy
- OpenCV
- TensorFlow / Keras

## ▶️ How to Run
(Add your code instructions here)

## 📷 Results
- Confusion matrix
- Training & validation curves

## 👤 Author
Mohammad Yousef
