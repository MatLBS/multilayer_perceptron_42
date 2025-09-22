import numpy as np
from toolkit_mlp.utils import plot_graphs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLP:
    def __init__(self, hidden_layer_sizes=(30, 30), learning_rate=0.01, n_epochs=1000, batch_size=32):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.train_loss_history = []
        self.valid_loss_history = []
        self.train_accuracy_history = []
        self.valid_accuracy_history = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        # return np.exp(x) / np.sum(np.exp(x))
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_exp_x

    def _binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _initialize_parameters(self, n_features):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [2]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def _feedforward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self._relu(z)
            activations.append(a)

        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z_output)
        y_pred = self._softmax(z_output)
        activations.append(y_pred)

        return activations, zs
    
    def _backpropagation(self, X_batch, y_batch, activations, zs):

        delta = activations[-1] - y_batch

        dW = (1/self.batch_size) * np.dot(activations[-2].T, delta)
        db = (1/self.batch_size) * np.sum(delta)
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db
        
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(zs[l])
            dW = (1/self.batch_size) * np.dot(activations[l].T, delta)
            db = (1/self.batch_size) * np.sum(delta)

            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * db

    def fit(self, X, y, X_valid, y_valid):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        print("x_train shape: ", X.shape)
        print("y_train shape: ", y.shape)
        print("x_valid shape: ", X_valid.shape)
        print("y_valid shape: ", y_valid.shape)

        for epoch in range(self.n_epochs):
            # shuffle datasets
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                activations, zs = self._feedforward(X_batch)
                self._backpropagation(X_batch, y_batch, activations, zs)

            activations, _ = self._feedforward(X)
            train_loss = self._binary_cross_entropy(y, activations[-1])
            self.train_loss_history.append(train_loss)
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(y, axis=1)
            self.train_accuracy_history.append(accuracy_score(y_true, y_pred))


            activations, _ = self._feedforward(X_valid)
            valid_loss = self._binary_cross_entropy(y_valid, activations[-1])
            self.valid_loss_history.append(valid_loss)
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(y_valid, axis=1)
            self.valid_accuracy_history.append(accuracy_score(y_true, y_pred))

            print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")

        plot_graphs(self.train_loss_history, self.valid_loss_history, self.train_accuracy_history, self.valid_accuracy_history)