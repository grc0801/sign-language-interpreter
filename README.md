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

## Folder Descriptions

- dataset: Contains the gesture image dataset used for training.
- model: Stores the trained CNN model (.h5) and the converted TFLite model (.tflite).
- src: Contains all source code files including gesture classification, hand tracking, and display handling.
- display: Contains OLED-related code and configuration.
- requirements.txt: Lists all required Python packages.
- train_model.py: Script to train the CNN model.
- convert_model.py: Script to convert the trained model to TensorFlow Lite format.
- main.py: The main executable script that runs the interpreter system.
- README.md: Documentation for the project.

--------------------------------------------------------------------------------

## Installation Instructions

Step 1: Clone the repository

    git clone https://github.com/grc0801/sign-language-interpreter.git
    cd sign-language-interpreter

Step 2: Install Python dependencies

    pip install -r requirements.txt

Or install them manually:

    pip install opencv-python mediapipe tensorflow pyttsx3 gTTS numpy

Step 3: If using Raspberry Pi, install espeak:

    sudo apt-get install espeak

--------------------------------------------------------------------------------

## How to Run

To run using a webcam:

    python main.py --mode webcam

To run using an IP camera stream:

    python main.py --mode ip --url http://your_ip_camera_stream

Ensure that the model file is placed in the "model" folder as "model.tflite".

--------------------------------------------------------------------------------

## Model Training (Optional)

To train your own model using your dataset:

1. Organize your images in subfolders by class inside the "dataset" folder.
2. Run the training script:

    python train_model.py

To convert the trained model to TensorFlow Lite format:

    python convert_model.py

--------------------------------------------------------------------------------

## Sample Output Description

- Captures hand gesture in real time
- Detects and classifies the gesture
- Displays the interpreted word on the OLED screen or monitor
- Speaks the word using a text-to-speech engine

--------------------------------------------------------------------------------

## Deployment on Raspberry Pi 5

1. Clone this repository on your Raspberry Pi.
2. Install all required Python packages.
3. Connect and configure the OLED display via I2C or SPI.
4. Plug in a webcam or use a Pi Camera module.
5. Run the main script:

    python main.py

--------------------------------------------------------------------------------

## Features

- Real-time recognition of hand gestures
- Works offline (with pyttsx3)
- Text display on screen or OLED
- Speech output for interpreted signs
- Optimized for portable edge deployment

--------------------------------------------------------------------------------

## Future Improvements

- Add support for dynamic gestures
- Expand dataset with more words and gestures
- Build a GUI for better user experience
- Add multilingual output support
- Further reduce model size for better Raspberry Pi performance

--------------------------------------------------------------------------------

## Acknowledgements

- MediaPipe for hand tracking
- TensorFlow for deep learning framework
- OpenCV for image and video handling
- pyttsx3 and gTTS for text-to-speech capabilities

--------------------------------------------------------------------------------

## License

This project is licensed under the MIT License.
