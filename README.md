# 🫁 PneumoVision

### AI-Powered Pneumonia Detection from Chest X-rays

**PneumoVision** is a deep learning project designed to classify chest X-ray images and identify whether they show signs of **pneumonia**.

The project combines **TensorFlow, MobileNetV2, Streamlit, and Grad-CAM** to create an interactive application that doesn't just provide a prediction, but also gives users a better understanding of the model's decision.

> ⚠️ **Disclaimer:** PneumoVision is an educational and research project. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice or diagnosis.

---

## 💡 Our Idea

The idea behind PneumoVision was to explore how **Artificial Intelligence and Computer Vision can be applied to medical imaging**.

Chest X-rays are commonly used by healthcare professionals to examine the lungs and identify abnormalities such as pneumonia. We wanted to build a system capable of learning visual patterns from chest X-ray images and using those learned patterns to classify new images.

However, we didn't want the project to simply output:

> **Pneumonia: YES/NO**

We wanted to make the model's prediction more understandable.

That's why PneumoVision includes:

* 🧠 Deep-learning-based image classification
* 📊 Prediction probability
* 📈 Visual probability bar
* 🔥 **Grad-CAM** visualization for model explainability
* 🖥️ Interactive Streamlit interface
* ℹ️ Model information and usage guidance
* ⚠️ Medical disclaimer

---

## 🧠 Why MobileNetV2?

For this project, we chose **MobileNetV2** as the foundation of our neural network.

MobileNetV2 is a convolutional neural network architecture designed to achieve a good balance between **accuracy and computational efficiency**.

Unlike very large architectures that can require substantial computational resources, MobileNetV2 is relatively lightweight while still being capable of learning meaningful visual features.

This made it a suitable choice for our project because we wanted to:

* Work with limited computational resources
* Train an image-classification model efficiently
* Keep the final application relatively lightweight
* Build a model that could eventually be used in an interactive application

### Transfer Learning

Rather than training a massive convolutional neural network completely from scratch, PneumoVision uses the concept of **transfer learning**.

A pretrained MobileNetV2 model provides a strong starting point for extracting visual features from images. The model is then adapted and trained for our specific task: **classifying chest X-rays into pneumonia and normal categories**.

This approach allows us to take advantage of features learned from large-scale image datasets while adapting the network to our medical-imaging problem.

---

## 🔥 Model Explainability with Grad-CAM

One of the important parts of PneumoVision is **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Deep learning models can sometimes behave like a black box: they provide an answer without clearly showing why they reached that conclusion.

Grad-CAM helps address this by creating a heatmap showing regions of an image that contributed strongly to the model's prediction.

In PneumoVision, the Grad-CAM visualization is overlaid onto the original chest X-ray so that users can visually inspect the areas that influenced the model.

This does **not** mean that the highlighted area proves the presence of pneumonia. It is simply a visualization of the regions that were influential to the model's prediction.

---

## 📊 Dataset

The chest X-ray data used in this project comes from a **real, publicly available dataset obtained from a valid source**.

The dataset contains chest X-ray images belonging to categories used for pneumonia classification.

The data was used for educational and research purposes to train and evaluate the model.

The dataset itself is **not artificially generated** for this project.

> **Dataset attribution:** Please refer to the dataset's original source and licensing terms before redistributing the images or dataset itself.

---

## ⚙️ How PneumoVision Works

The overall workflow can be summarized as:

```text
Chest X-ray Image
        ↓
Image Preprocessing
        ↓
MobileNetV2
        ↓
Fine-tuned Classification Model
        ↓
Prediction
        ↓
Probability
        ↓
Grad-CAM
        ↓
Streamlit Interface
```

### 1. Image Input

The user uploads a chest X-ray through the Streamlit application.

### 2. Preprocessing

The image is resized and processed into the format expected by the trained model.

### 3. Prediction

The processed image is passed through the trained MobileNetV2-based neural network.

### 4. Probability

The model produces a prediction along with its probability.

### 5. Grad-CAM

Grad-CAM is generated to visualize the image regions that contributed to the prediction.

### 6. Streamlit Interface

The prediction, probability, visualization, model information, and disclaimer are presented through an interactive web interface.

---

## 🖥️ Application

The final application was built using **Streamlit**, allowing the trained model to be presented through a simple interactive interface without requiring users to interact directly with Python code.

The application allows users to:

* Upload a chest X-ray
* Generate a prediction
* View prediction probability
* View a probability visualization
* Generate a Grad-CAM heatmap
* View information about the model
* Understand the limitations of the application

---

## 🧪 Development Process

PneumoVision wasn't built in a single script or a single attempt.

The project went through many stages:

**Dataset → preprocessing → model building → training → evaluation → visualization → debugging → improvement → deployment**

During development, multiple experiments were performed to understand:

* Image preprocessing
* Dataset organization
* CNN-based classification
* Transfer learning
* MobileNetV2
* Model training
* Validation performance
* Prediction behavior
* Confusion matrices
* Grad-CAM
* Streamlit deployment
* Model integration

As a result, the repository contains several files from different stages of development.

Rather than removing every experimental file, they are retained as part of the project's development history.

This makes PneumoVision not only a final application, but also a record of the **learning and engineering process behind it**.

---

## 🛠️ Technologies Used

| Technology             | Purpose                              |
| ---------------------- | ------------------------------------ |
| **Python**             | Core programming language            |
| **TensorFlow / Keras** | Deep learning and model training     |
| **MobileNetV2**        | CNN architecture / transfer learning |
| **NumPy**              | Numerical operations                 |
| **OpenCV**             | Image processing                     |
| **Matplotlib**         | Visualization                        |
| **Scikit-learn**       | Model evaluation                     |
| **Streamlit**          | Interactive web application          |
| **Grad-CAM**           | Model explainability                 |

---

## 🚀 Running the Application

Clone or download the project and install the required dependencies.

Then navigate to the project directory and run:

```bash
streamlit run app.py
```

Streamlit will launch the application locally in your browser.

---

## 🎯 Project Goals

The main goals of PneumoVision were to:

1. Build a practical deep learning image-classification project.
2. Understand how transfer learning works.
3. Explore MobileNetV2 in a real-world classification task.
4. Work with an authentic medical-imaging dataset.
5. Evaluate model predictions rather than relying only on training accuracy.
6. Explore explainable AI through Grad-CAM.
7. Deploy a trained machine-learning model through Streamlit.
8. Gain practical experience with the complete ML project lifecycle.

---

## 📚 What We Learned

Building PneumoVision provided hands-on experience with the complete process of developing a machine-learning application:

**Data → Preprocessing → Model → Training → Validation → Evaluation → Explainability → Deployment**

It also demonstrated an important reality of machine-learning projects:

> The final result is usually the product of many experiments, mistakes, failed approaches, debugging sessions, and improvements.

The numerous files in this repository are therefore part of that journey.

---

## ⚠️ Medical Disclaimer

PneumoVision is **not a medical device** and has not been developed or validated for clinical diagnosis.

Predictions generated by the model may be incorrect and should not be used to make medical decisions.

If you have concerns about a chest X-ray or your health, consult a qualified healthcare professional.

---

## 👨‍💻 Project

**Project:** PneumoVision
**Domain:** Artificial Intelligence / Machine Learning / Computer Vision
**Task:** Chest X-ray Classification
**Model:** MobileNetV2 with Transfer Learning
**Interface:** Streamlit
**Explainability:** Grad-CAM

---

### ⭐ Final Note

PneumoVision started as an experiment to understand how deep learning could be applied to chest X-ray images and evolved into a complete machine-learning application.

From working with the dataset and training the model to evaluating predictions, implementing Grad-CAM, debugging the application, and finally deploying everything through Streamlit, the project represents a hands-on exploration of the **end-to-end machine-learning workflow**.

**Built through experimentation, trial and error, and a lot of debugging.**
