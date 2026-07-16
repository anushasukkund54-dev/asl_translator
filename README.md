# asl_translator
A computer vision project that translates hand signs into spoken audio in real-time.
# ASL Translator

A machine learning project designed to recognize American Sign Language (ASL) gestures in real-time using computer vision.

## 🚀 Overview
This application uses a trained model to translate ASL manual alphabet signs captured via webcam into text. It includes a custom data processing pipeline for training and testing.

## 📁 Project Structure
 *live_translate.py: The main script used to perform real-time translation by capturing hand signs via your webcam.
 *train.py: The script responsible for training the machine learning model.
 *test.py: A utility script used for evaluating the model and testing its accuracy.
 *create_csv.py: A helper script designed to manage and structure your dataset files.
 *asl_model.h5: The saved file containing your trained machine learning model.
 *final_dataset.csv: The processed dataset file used to train the model.

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