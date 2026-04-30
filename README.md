
# 🧠 NeuroScan AI – Brain Tum Detection Using Deep CNN

## AI-Based Early Brain Tumor Detection using MRI Image Analysis

NeuroScan AI is a healthcare-focused AI/ML project developed for early brain tumor detection using Deep Learning and MRI scan classification.

This system helps classify brain MRI images into two categories:

* Tumor Detected
* No Tumor Detected

The project uses an EfficientNetB0-based Convolutional Neural Network (CNN) model for image classification and is deployed using Streamlit as a professional healthcare-style web application.

Brain tumors are critical medical conditions where early diagnosis can significantly improve treatment outcomes.

Traditional diagnosis depends heavily on radiologists manually analyzing MRI scans, which can be time-consuming and may delay treatment.

The objective of this project is to build an AI-powered diagnostic assistant that helps in early tumor screening by automatically analyzing MRI brain scans and predicting tumor presence.

Manual MRI analysis requires expert radiologists and significant time.

Challenges include:

* Delayed diagnosis
* Human error possibility
* Limited specialist availability
* Need for faster preliminary screening

This project aims to support early diagnosis using Artificial Intelligence.

---
# 🧠 Technologies Used

* Python

## Frameworks & Libraries

* TensorFlow
* Keras
* Streamlit
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* Pillow

## Deep Learning Model

* EfficientNetB0 + Custom CNN Layers

## Deployment

* Streamlit Web Application
# 📂 Dataset Structure

```text
dataset/
 ├── train/
 │   ├── tumor/
 │   └── no_tumor/
 │
 └── test/
     ├── tumor/
     └── no_tumor/
```

Dataset contains MRI brain scan images divided into:

* Tumor images
* No Tumor images

The dataset is split into:

* Training set (for model learning)
* Testing set (for model validation)

---

# 🤖 Why CNN?

CNN (Convolutional Neural Network) is best suited for image classification tasks.

It automatically detects:

* Edges
* Textures
* Tumor patterns
* Abnormal regions
* Hidden visual features

Unlike traditional Machine Learning, CNN does not require manual feature extraction.

---

# ⚙ Model Architecture

```text
EfficientNetB0
↓
GlobalAveragePooling2D
↓
Dense(128, activation='relu')
↓
Dropout(0.5)
↓
Dense(1, activation='sigmoid')
```

## Why These Layers?

### EfficientNetB0

Used as the base pre-trained model for powerful feature extraction.

### Dense Layer

Learns deeper classification patterns.

### Dropout

Prevents overfitting and improves generalization.

### Sigmoid Layer

Used for binary classification:

* Tumor
* No Tumor

---

# 🔄 Data Preprocessing

Before training:

* Images resized to 224 × 224
* Grayscale conversion
* Pixel normalization using `rescale = 1./255`

This improves training stability and model performance.

---

# 🔁 Data Augmentation

To reduce overfitting:

* Rotation
* Zoom
* Horizontal Flip

This improves model generalization on unseen MRI scans.

---

# 🏋 Model Training

Model compiled using:

```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
```

## Why?

### Adam Optimizer

Fast and efficient learning.

### Binary Crossentropy

Best for binary classification tasks.

### Accuracy

Used to evaluate model performance.

Training performed using:

```python
model.fit()
```

---

# 💾 Model Saving

After successful training:

```python
model.save("model/brain_tumor_model.h5")
```

This saves the final trained AI model for deployment.

---

# 🌐 Streamlit Deployment

The trained model is deployed using Streamlit to create a professional healthcare dashboard.

## Features

* Upload MRI image
* Run diagnostic prediction
* Tumor / No Tumor result
* Confidence Score
* Progress Bar Visualization
* Clinical Recommendation Box
* Hospital-style Interface

---

# 📊 Output Example

## Positive Finding

```text
Brain Tumor Detected
Confidence Score: 92.45%
Clinical Recommendation:
Immediate consultation with neurologist advised
```

## Negative Finding

```text
No Tumor Detected
Confidence Score: 89.30%
Clinical Recommendation:
Continue regular health monitoring
```

---

# 🎯 Key Features

* Deep CNN model training using TensorFlow
* EfficientNetB0 transfer learning
* Real-time MRI prediction
* Professional healthcare UI
* Confidence score visualization
* Clinical recommendation support
* GitHub portfolio-ready project

---

# 🔮 Future Scope

Future improvements include:

* Multi-class tumor detection
* Tumor segmentation using Grad-CAM
* Voice assistant integration
* PDF medical report generation
* Cloud deployment
* Doctor dashboard system
* Hospital integration

---

# ▶ How to Run the Project

## Step 1: Install Dependencies

```bash
pip install tensorflow streamlit numpy matplotlib opencv-python scikit-learn pillow
```

## Step 2: Train Model

```bash
python train.py
```

## Step 3: Run Streamlit App

```bash
streamlit run app.py
```

## Step 4: Open Browser

```text
http://localhost:8501
```

Upload MRI image and run prediction.

Harshita P
Final Year Electronics and Communication Engineering Student
AI/ML Intern – SIC

# ⭐ Final Note

This project demonstrates the practical application of Artificial Intelligence in healthcare using Deep Learning and medical image classification.

It combines:

* AI + Healthcare
* CNN + MRI Analysis
* Model Training + Real-world Deployment

