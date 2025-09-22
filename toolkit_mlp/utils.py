from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _train_test_split(arr, ratio=0.8):
    np.random.shuffle(arr)
    total_rows = arr.shape[0]
    train_size = int(total_rows * ratio)

    # Split data into test and train
    train = arr[:train_size]
    test = arr[train_size:]

    return train, test


def preprocess_data(df):
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X = df.drop(columns=[0, 1], axis=1)
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    return X


def draw_histogram(file: str) -> None:
    df = pd.read_csv(file, header=None)

    benign = df[df[1] == 'B']
    malignant = df[df[1] == 'M']
    columns = df.columns[2:]

    feature_list = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]
    feat_type = ["mean", "standard error", "largest"]

    fig, ax = plt.subplots(6, 5, figsize=(14, 10))
    k = 0

    for i in range(6):
        for j in range(5):
            if k < len(columns):
                ax[i][j].hist(benign[columns[k]], color='blue', bins=20, alpha=0.4)
                ax[i][j].hist(malignant[columns[k]], color='red', bins=20, alpha=0.4)
                ax[i][j].set_title(f"{feature_list[k % 10]} {feat_type[k // 10]}")
            else:
                ax[i][j].axis('off')
            k += 1
    plt.tight_layout()
    plt.show()


def plot_graphs(train_loss_history, valid_loss_history, train_accuracy_history, valid_accuracy_history):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_loss_history, label="Train Loss")
    # ax[0].plot(valid_loss_history, label="Valid Loss")
    ax[0].set_title("Loss Function Convergence")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(train_accuracy_history, label="Train Accuracy")
    ax[1].plot(valid_accuracy_history, label="Valid Accuracy")
    ax[1].set_title("Accuracy Function Convergence")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    plt.show()