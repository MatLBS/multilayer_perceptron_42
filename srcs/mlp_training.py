import pandas as pd
import sys
import os
import numpy as np
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_mlp.utils import _train_test_split, preprocess_data
from toolkit_mlp.mlp_utils import MLP_SGD, MLP_Adam


def train_model(args: argparse.Namespace):
    df = pd.read_csv(args.file, header=None)

    X = preprocess_data(df)
    y = df[1].values
    y = np.array([[1, 0] if label == 'M' else [0, 1] for label in y])

    X_train, X_valid, y_train, y_valid = _train_test_split(X, y)

    model = MLP_SGD(
        hidden_layer_sizes=args.layer,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )

    model.fit(X_train, y_train, X_valid, y_valid)
    model.save_model(X_train)


def main():
    try:
        assert len(sys.argv) > 2, "You must provide at least the dataset file path as first argument"
        assert os.path.exists(sys.argv[2]), "The file does not exists"

        parser = argparse.ArgumentParser()
        parser.add_argument('--file', type=str)
        parser.add_argument('--layer', type=int, nargs='+')
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--learning_rate', type=float)
        args = parser.parse_args()
        train_model(args)
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
