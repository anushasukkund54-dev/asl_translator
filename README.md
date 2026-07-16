# asl_translator
A computer vision project that translates hand signs into spoken audio in real-time.
# ASL Translator

A machine learning project designed to recognize American Sign Language (ASL) gestures in real-time using computer vision.

## 🚀 Overview
This application uses a trained model to translate ASL manual alphabet signs captured via webcam into text. It includes a custom data processing pipeline for training and testing.

## 📁 Project Structure
 live_translate.py: Main script for real-time webcam inference.
 train.py: Model training script.
 test.py: Model evaluation and accuracy testing.
 create_csv.py: Utility to manage/structure dataset files.
 asl_model.h5: The trained machine learning model.
 final_dataset.csv: Processed dataset for model training.
 
## 🛠 Prerequisites
Ensure you have the following installed:
* Python 3.x
* OpenCV
* TensorFlow/Keras
* NumPy & Pandas

## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/anushasukkund54-dev/asl_translator.git](https://github.com/anushasukkund54-dev/asl_translator.git)