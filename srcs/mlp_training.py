import pandas as pd
import matplotlib.pyplot as plt
from toolkit_mlp.utils import _train_test_split, preprocess_data


def train_model(file: str) -> None:
    df = pd.read_csv(file, header=None)

    X = preprocess_data(df)
    y = df[1].values
    X_train, X_test = _train_test_split(X)
    y_train, y_test = _train_test_split(y)

    print("Training set size:", X_train.shape, y_train.shape)
    print("Test set size:", X_test.shape, y_test.shape)


def main():
    df = pd.read_csv("data.csv")
    # print(df.describe())
    train_model("data.csv")

if __name__ == "__main__":
    main()