# CodeAlpha_EmotionRecognitionFromSpeech
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
# 🎙️ Speech Emotion Recognition (SER) using CRNN

An end-to-end Deep Learning and Audio Processing application designed to recognize human emotions from speech signals using a **Convolutional Recurrent Neural Network (CRNN)** built with TensorFlow/Keras and deployed via an interactive **Gradio** web application.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Architecture](#-dataset-architecture)
- [Model Architecture](#-model-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Model Training & Notebook Execution](#1-model-training--notebook-execution)
  - [2. Interactive Web Application](#2-interactive-web-application)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 📌 Project Overview

Speech Emotion Recognition (SER) plays a crucial role in Human-Computer Interaction (HCI), allowing intelligent systems to analyze audio signals and infer human emotional states. 

This repository provides a complete pipeline that:
1. Extracts spatial-temporal audio features (MFCCs and Mel Spectrograms) from raw audio recordings.
2. Trains a hybrid **Convolutional Recurrent Neural Network (CRNN)** to capture both local frequency patterns and temporal sequence dynamics.
3. Provides a live web interface built with **Gradio** to allow users to upload or record speech audio and get real-time emotion predictions.

---

## ✨ Key Features

- **Audio Feature Extraction**: Transforms raw `.wav` speech signals into 2D Log-Mel Spectrogram representations using `librosa`.
- **CRNN Architecture**: Combines 2D Convolutional layers (for spatial feature extraction) with Recurrent/LSTM units (for temporal sequence modeling).
- **Interactive Web App**: Includes a ready-to-run [Gradio interface](gradio_emotion_app.py) for real-time audio testing via microphone recording or file uploads.
- **Exported Model**: Comes pre-packaged with the trained model artifact (`speech_emotion_crnn_model.keras`) for instant inference.

---

## 📊 Dataset Architecture

This project is trained and evaluated on the **[Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)** dataset.

- **Audio Format**: 200 target words spoken by two female actors (aged 26 and 64).
- **Class Categories (7 Emotions)**:
  - 😃 **Happy**
  - 😮 **Surprise**
  - 😡 **Anger**
  - 😨 **Fear**
  - 😒 **Disgust**
  - 😢 **Sad**
  - 😐 **Neutral**

---

## 🧠 Model Architecture

The model uses a hybrid **Convolutional Recurrent Neural Network (CRNN)** topology to process 2D audio representations:

1. **Input Layer**: Takes 2D log-Mel spectrogram feature matrices generated from standard 3-second audio clips.
2. **Convolutional Blocks**: Multiple `Conv2D` layers paired with `BatchNormalization`, `ReLU` activations, and `MaxPooling2D` to extract spatial acoustic features.
3. **Recurrent Blocks**: Stacked `Bidirectional(LSTM)` or `GRU` layers to capture context and long-term temporal dependencies across speech frames.
4. **Dense Output**: Fully connected `Dense` layers ending with a 7-class `Softmax` output layer to produce emotion class probabilities.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package installer

### 1. Clone the Repository
```bash
git clone [https://github.com/shashwat123u/CodeAlpha_EmotionRecognitionFromSpeech.git](https://github.com/shashwat123u/CodeAlpha_EmotionRecognitionFromSpeech.git)
cd CodeAlpha_EmotionRecognitionFromSpeech
