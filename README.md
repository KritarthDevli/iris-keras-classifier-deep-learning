# Iris Keras Classifier

This project implements a neural network classifier for the Iris dataset using TensorFlow and Keras. The model predicts flower species based on sepal and petal measurements using a fully connected artificial neural network (ANN).

## Project Overview
- Dataset: Iris (150 samples, 4 features, 3 classes)
- Model: Feedforward ANN with normalization and dense layers
- Output: Softmax probabilities for each species
- Evaluation: Training accuracy and incorrect predictions
- Visualization: Training loss vs epoch

## Project Structure
iris-keras-classifier/
├─ iris_project.py
├─ requirements.txt
├─ README.md
├─ outputs/
│ ├─ loss.png
│ └─ model.keras

## Requirements
- Python 3.9+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
How to Run
python iris_project.py


This will:

Load the Iris dataset from the web

One-hot encode species labels

Train the neural network

Plot and save the loss curve

Save the trained model

Print accuracy and incorrect prediction indices

Model Architecture

Input: 4 normalized features

Hidden Layer 1: 8 neurons (sigmoid activation)

Hidden Layer 2: 8 neurons (ReLU activation)

Output Layer: 3 neurons (softmax activation)

Mathematically, the model represents:
ANN: ℝ⁴ → 𝕂³

Output Files

outputs/loss.png – training loss curve

outputs/model.keras – trained Keras model
