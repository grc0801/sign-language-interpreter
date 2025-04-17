# Real-Time Sign Language Interpreter  
**Using Deep Learning and Raspberry Pi for the Hearing and Speech Impaired**

---

## 📌 Overview / Description

This project presents a real-time Sign Language Interpreter designed to assist individuals who are hearing and speech impaired. It captures hand gestures using a webcam or IP camera, processes them using deep learning and MediaPipe, and outputs the interpreted sign as both text and speech. The system is deployed on a Raspberry Pi 5, making it portable and efficient for real-world use.

---

## 🎯 Objective

The primary objective of this project is to develop a real-time sign language interpretation system that aids communication for individuals who are hearing and speech impaired. The system uses deep learning models, MediaPipe hand tracking, and Raspberry Pi to detect and classify hand gestures in real time, and converts them into textual and audio output.

This project aims to:

- Recognize static hand gestures using Convolutional Neural Networks (CNNs)
- Enable real-time gesture classification with high accuracy
- Display the predicted word on a screen (OLED/Monitor)
- Provide speech output using text-to-speech engines
- Deploy the model on Raspberry Pi 5 for a portable, low-power solution

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow & TensorFlow Lite**
- **OpenCV** – Video capture
- **MediaPipe** – Hand tracking with 21 landmarks
- **pyttsx3** / **gTTS** – Text-to-speech conversion
- **Raspberry Pi 5** – Deployment hardware
- **OLED Display** – For text output

---

## 🗂️ Folder Structure


---

## 💻 Installation Instructions

### 1. Clone the repository
```bash
git clone https://github.com/grc0801/sign-language-interpreter.git
cd sign-language-interpreter

pip install -r requirements.txt

python main.py’’’

---

##



