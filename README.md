# PythonEmotionDetecter

## Description
This project uses DeepFace and OpenCV to perform real-time emotion detection through your webcam. The application analyses each video frame to detect facial emotions and visually highlights detected faces with a box and the emotion label. It’s an intuitive demo of AI-based emotion recognition.

[![Demo Video](https://img.youtube.com/vi/LmL2S5u2oRg/0.jpg)](https://youtu.be/LmL2S5u2oRg)

## Features

- **Real-Time Analysis:** Captures live video from your webcam.
- **Emotion Recognition:** Uses DeepFace to analyze facial emotions.
- **Visual Feedback:** Draws a green rectangle around detected faces and overlays the dominant emotion.
- **User-Friendly:** Press `q` at any time to exit the application.

## Prerequisites

- Python 3.11 (or a compatible version)
- Required Python packages:
  - [OpenCV-Python](https://pypi.org/project/opencv-python/)
  - [DeepFace](https://pypi.org/project/deepface/)

## Installation

1. **Clone or Download the Repository**

   Clone this repository or download the source code to your local machine.

2. **Navigate to the Project Directory**

   Open your terminal (or command prompt) and change to the project directory:

   ```bash
   cd /path/to/PythonEmotionDetecter
