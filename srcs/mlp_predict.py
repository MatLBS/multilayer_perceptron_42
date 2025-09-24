import sys
import os
import numpy as np
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_mlp.utils import _train_test_split, preprocess_data
from toolkit_mlp.mlp_utils import MLP
from sklearn.metrics import accuracy_score


def predict_model(file_weights: str, file_topology: str, file_data: str):
    data_weights = np.load(file_weights)
    data_topology = json.load(open(file_topology))

    df = pd.read_csv(file_data, header=None)
    X = preprocess_data(df)
    y = df[1].values
    y = np.array([[1, 0] if label == 'M' else [0, 1] for label in y])

    X_train, X_valid, y_train, y_valid = _train_test_split(X, y)

    model = MLP(
        hidden_layer_sizes=data_topology['hidden_layer_sizes'],
        learning_rate=data_topology['learning_rate'],
        n_epochs=data_topology['n_epochs'],
        batch_size=data_topology['batch_size'],
        output_size=data_topology['output_size']
    )

    length = len(data_weights.files)
    model.weights = [data_weights[f'arr_{i}']
                     for i in range(length // 2)]
    model.biases = [data_weights[f'arr_{i}']
                    for i in range(length // 2, length)]
    y_pred = model.predict(X_valid)
    y_true = np.argmax(y_valid, axis=1)

    print(accuracy_score(y_true, y_pred))


def main():
    try:
        assert len(sys.argv) == 4, "You must provide the weights,"
        "topology and data files paths"
        assert os.path.exists(sys.argv[1]), "The weights file does not exist"
        assert os.path.exists(sys.argv[2]), "The topology file does not exist"
        assert os.path.exists(sys.argv[3]), "The data file does not exist"
        predict_model(sys.argv[1], sys.argv[2], sys.argv[3])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
