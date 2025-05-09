# Real-Time Sign Language Interpreter  
Using Deep Learning and Raspberry Pi for the Hearing and Speech Impaired

--------------------------------------------------------------------------------

## Overview / Description

This project presents a real-time Sign Language Interpreter designed to assist individuals who are hearing and speech impaired. It captures hand gestures using a webcam or IP camera, processes them using deep learning and MediaPipe, and outputs the interpreted sign as both text and speech. The system is deployed on a Raspberry Pi 5, making it portable and efficient for real-world use.

--------------------------------------------------------------------------------

## Objective

The primary objective of this project is to develop a real-time sign language interpretation system that aids communication for individuals who are hearing and speech impaired. The system uses deep learning models, MediaPipe hand tracking, and Raspberry Pi to detect and classify hand gestures in real time, and converts them into textual and audio output.

This project aims to:

- Recognize static hand gestures using Convolutional Neural Networks (CNNs)
- Enable real-time gesture classification with high accuracy
- Display the predicted word on a screen (OLED/Monitor)
- Provide speech output using text-to-speech engines
- Deploy the model on Raspberry Pi 5 for a portable, low-power solution

--------------------------------------------------------------------------------

## Tech Stack

- Python 3
- TensorFlow and TensorFlow Lite
- OpenCV – Video capture
- MediaPipe – Hand tracking with 21 landmarks
- pyttsx3 or gTTS – Text-to-speech conversion
- Raspberry Pi 5 – Deployment hardware
- OLED Display – For text output

--------------------------------------------------------------------------------

## Folder Structure

sign-language-interpreter/
│
├── dataset/               --> Contains training images
├── model/                 --> Trained .h5 and .tflite models
├── src/                   --> Main source code
│   ├── hand_tracker.py    --> Hand landmark detection using MediaPipe
│   ├── classifier.py      --> CNN-based gesture classifier
│   ├── tts.py             --> Text-to-speech module
│   └── display.py         --> OLED display handling
├── display/               --> OLED configuration and scripts
├── requirements.txt       --> List of Python dependencies
├── train_model.py         --> Script to train the CNN model
├── convert_model.py       --> Script to convert to TFLite
├── main.py                --> Main execution script
└── README.md              --> Project documentation

--------------------------------------------------------------------------------

## Installation Instructions

Step 1: Clone the repository

    git clone https://github.com/grc0801/sign-language-interpreter.git
    cd sign-language-interpreter

Step 2: Install Python dependencies

    pip install -r requirements.txt

Alternatively, manually install:

    pip install opencv-python mediapipe tensorflow pyttsx3 gTTS numpy

Step 3: For Raspberry Pi (Optional):

    sudo apt-get install espeak

--------------------------------------------------------------------------------

## How to Run

To run with webcam:

    python main.py --mode webcam

To run with IP camera:

    python main.py --mode ip --url http://your_ip_camera_stream

Ensure that the model file is present in the 'model/' folder as 'model.tflite'.

--------------------------------------------------------------------------------

## Model Training (Optional)

To train your own model:

1. Place gesture images inside 'dataset/' with subfolders per class
2. Run:

    python train_model.py

To convert the trained model to TensorFlow Lite format:

    python convert_model.py

--------------------------------------------------------------------------------

## Sample Output

Sample system behavior:

- Captures hand gesture
- Detects and classifies gesture
- Displays recognized word on screen
- Outputs spoken word via speaker

--------------------------------------------------------------------------------

## Deployment on Raspberry Pi 5

1. Clone the repository on Raspberry Pi
2. Install required libraries (use TFLite runtime if needed)
3. Connect OLED display and configure via display.py
4. Attach USB camera or Pi Camera module
5. Run the project:

    python main.py

--------------------------------------------------------------------------------

## Features

- Real-time gesture recognition
- Offline functionality using pyttsx3
- Portable deployment on Raspberry Pi 5
- Text output via monitor or OLED
- Audio output for interpreted sign

--------------------------------------------------------------------------------

## Future Improvements

- Add support for dynamic gestures (gesture sequences)
- Expand dataset with more vocabulary
- Add graphical user interface (GUI)
- Multilingual audio output support
- Optimize model size for faster inference on edge devices

--------------------------------------------------------------------------------

## Acknowledgements

- MediaPipe for hand tracking
- TensorFlow for model training and inference
- OpenCV for image and video processing
- pyttsx3 and gTTS for text-to-speech functionality

--------------------------------------------------------------------------------

## License

This project is licensed under the MIT License.
