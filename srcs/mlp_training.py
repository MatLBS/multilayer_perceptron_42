import pandas as pd
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_mlp.utils import _train_test_split, preprocess_data
from toolkit_mlp.mlp_utils import MLP


def train_model(file: str) -> None:
    df = pd.read_csv(file, header=None)

    X = preprocess_data(df)
    y = df[1].values
    y = np.array([[1, 0] if label == 'M' else [0, 1] for label in y])
    X_train, X_valid = _train_test_split(X)
    y_train, y_valid = _train_test_split(y)

    model = MLP(
        hidden_layer_sizes=(30, 30),
        learning_rate=0.01,
        n_epochs=1000,
        batch_size=32
    )

    model.fit(X_train, y_train, X_valid, y_valid)

    # for idx, weigth in enumerate(model.weights):
    #     print("-------------------------------------------")
    #     # print(weigth)
    #     data = {
    #         f"weigths to layer {idx+1}": weigth
    #     }

    np.savez('mlp_weights.npz', *model.weights, *model.biases)

    topology = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'input_size': X_train.shape[1],
        'output_size': model.output_size,
        'activation': 'relu',
        'activation_output': 'softmax',

    }
    with open('mlp_topology.json', 'w') as f:
        json.dump(topology, f)


def main():
    assert len(sys.argv) == 2, "You must provide the dataset file path"
    assert os.path.exists(sys.argv[1]), "The file does not exists"
    train_model(sys.argv[1])


if __name__ == "__main__":
    main()
