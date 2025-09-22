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


def plot_losses(train_loss_history, valid_loss_history):
    plt.plot(train_loss_history, label="Train Loss")
    # plt.plot(valid_loss_history, label="Valid Loss")
    plt.title("Loss Function Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()