"""
Iris Keras Classifier - Single File Project

- Loads Iris data from: http://raptor.kent.ac.uk/~ds756/Data/iris.csv
- One-hot encodes Species
- Builds a small ANN (Normalization + Dense + Dense + Softmax)
- Trains for N_EPOCHS
- Plots loss curve
- Predicts classes, prints accuracy + incorrect indices
- Saves outputs:
    outputs/loss.png
    outputs/model.keras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

IRIS_URL = "http://raptor.kent.ac.uk/~ds756/Data/iris.csv"


# -----------------------------
# Data
# -----------------------------
def load_iris(url: str = IRIS_URL) -> pd.DataFrame:
    return pd.read_csv(url)


def make_xy(df: pd.DataFrame):
    # X: first 4 columns
    X = df.iloc[:, 0:4].values

    # Y: one-hot encode Species
    enc = OneHotEncoder(sparse_output=False)
    Y = enc.fit_transform(df.loc[:, ["Species"]])

    return X, Y, enc


# -----------------------------
# Model
# -----------------------------
def build_model(input_dim: int = 4, hidden_units: int = 8, n_classes: int = 3):
    model = Sequential()

    # Normalization (matches your notebook approach)
    model.add(Normalization(input_shape=[input_dim]))

    # Hidden layers
    model.add(Dense(hidden_units, input_dim=input_dim, activation="sigmoid"))
    model.add(Dense(hidden_units, activation="relu"))

    # Output layer (3 classes)
    model.add(Dense(n_classes, activation="softmax", name="Prediction"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# -----------------------------
# Plot
# -----------------------------
def plot_loss(history, out_path: str | None = None):
    plt.figure()
    plt.plot(history.history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    # Load data
    iris = load_iris()
    print(iris.info())
    print(iris.describe())

    # Prepare X, Y
    X, Y, _ = make_xy(iris)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Build model
    model = build_model(input_dim=X.shape[1], n_classes=Y.shape[1])
    model.summary()

    # Train
    N_EPOCHS = 1000
    history = model.fit(X, Y, verbose=False, epochs=N_EPOCHS)

    # Save outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Plot loss
    plot_loss(history, out_path="outputs/loss.png")

    # Predict
    Y_hat = model.predict(X, verbose=0)  # 150x3
    K_hat = np.argmax(Y_hat, axis=1)

    # Compare with ground truth
    K_true = np.argmax(Y, axis=1)
    results = (K_hat == K_true)

    incorrect_idx = np.where(results == False)[0]  # indices where wrong
    accuracy = np.mean(results)

    print("Incorrect indices:", incorrect_idx.tolist())
    print("Accuracy:", accuracy)

    # Save model
    model.save("outputs/model.keras")
    print("Saved: outputs/loss.png and outputs/model.keras")


if __name__ == "__main__":
    main()
