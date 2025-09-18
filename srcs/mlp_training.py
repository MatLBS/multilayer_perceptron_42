import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_mlp.utils import _train_test_split, preprocess_data
from toolkit_mlp.mlp_utils import MLP


def train_model(file: str) -> None:
    df = pd.read_csv(file, header=None)

    X = preprocess_data(df)
    y = df[1].values
    # y = np.array([[1, 0] if label == 'B' else [0, 1] for label in y])
    X_train, X_test = _train_test_split(X)
    y_train, y_test = _train_test_split(y)

    # print("Training set size:", X_train.shape, y_train.shape)
    # print("Test set size:", X_test.shape, y_test.shape)

    model = MLP(
        hidden_layer_sizes=(30, 30, ),
        learning_rate=0.001,
        n_epochs=1000,
        batch_size=32
    )

    model.fit(X_train, y_train)




def main():
    assert len(sys.argv) == 2, "You must provide the dataset file path"
    assert os.path.exists(sys.argv[1]), "The file does not exists"
    df = pd.read_csv(sys.argv[1])
    # print(df.describe())
    train_model(sys.argv[1])


if __name__ == "__main__":
    main()