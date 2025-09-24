import pandas as pd
import sys
import os
import numpy as np
from toolkit_mlp.utils import _train_test_split, preprocess_data
from toolkit_mlp.mlp_utils import MLP
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def train_model(file: str) -> None:
    df = pd.read_csv(file, header=None)

    X = preprocess_data(df)
    y = df[1].values
    y = np.array([[1, 0] if label == 'M' else [0, 1] for label in y])

    X_train, X_valid, y_train, y_valid = _train_test_split(X, y)

    model = MLP(
        hidden_layer_sizes=(30, 30),
        learning_rate=0.02,
        n_epochs=1000,
        batch_size=32
    )

    model.fit(X_train, y_train, X_valid, y_valid)
    model.save_model(X_train)


def main():
    try:
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        train_model(sys.argv[1])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
