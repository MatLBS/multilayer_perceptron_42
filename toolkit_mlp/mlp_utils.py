import numpy as np
import json
from colorama import Fore, Style, init
from toolkit_mlp.utils import plot_graphs
from sklearn.metrics import (accuracy_score,
                              precision_score,
                              recall_score,
                              f1_score)


class MLP_SGD:
    def __init__(self, hidden_layer_sizes=(24, 24, 24), learning_rate=0.01,
                 n_epochs=1000, batch_size=32, output_size=2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.no_improve = 0
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
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def _binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _early_stopping(self):
        if len(self.valid_loss_history) > 2 and \
           (self.valid_loss_history[-1] > self.valid_loss_history[-2]):
            self.no_improve += 1
        else:
            self.no_improve = 0
        if self.no_improve > 5:
            print("Early stopping")
            return True
        return False

    def _initialize_parameters(self, n_features):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self.output_size]
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

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            activations, _ = self._feedforward(X_valid)
            valid_loss = self._binary_cross_entropy(y_valid, activations[-1])
            self.valid_loss_history.append(valid_loss)
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(y_valid, axis=1)
            self.valid_accuracy_history.append(accuracy_score(y_true, y_pred))

            if self._early_stopping():
                break

            print(f"Epoch {epoch+1}/{self.n_epochs}, "
                f"{Fore.YELLOW}Train Loss: {train_loss:.4f}{Style.RESET_ALL}, "
                f"{Fore.CYAN}Valid Loss: {valid_loss:.4f}{Style.RESET_ALL}, "
                f"{Fore.MAGENTA}Precision: {precision:.2f}{Style.RESET_ALL}, "
                f"{Fore.RED}Recall: {recall:.2f}{Style.RESET_ALL}, "
                f"{Fore.GREEN}F1: {f1:.2f}{Style.RESET_ALL}")

        plot_graphs(self.train_loss_history, self.valid_loss_history, self.train_accuracy_history, self.valid_accuracy_history)

    def predict(self, X):
        activations, _ = self._feedforward(X)
        return np.argmax(activations[-1], axis=1)

    def save_model(self, X_train):
        topology = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'input_size': X_train.shape[1],
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'activation': 'relu',
            'activation_output': 'softmax',
        }
        np.savez('mlp_weights.npz', *self.weights, *self.biases)
        with open('mlp_topology.json', 'w') as f:
            json.dump(topology, f)


class MLP_Adam:
    def __init__(self, hidden_layer_sizes=(24, 24, 24), learning_rate=0.001,
                 n_epochs=1000, batch_size=32, output_size=2, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.no_improve = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.weights = []
        self.biases = []

        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        self.train_loss_history = []
        self.valid_loss_history = []
        self.train_accuracy_history = []
        self.valid_accuracy_history = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def _binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _early_stopping(self):
        if len(self.valid_loss_history) > 2 and \
           (self.valid_loss_history[-1] > self.valid_loss_history[-2]):
            self.no_improve += 1
        else:
            self.no_improve = 0
        if self.no_improve > 5:
            print("Early stopping")
            return True
        return False

    def _initialize_parameters(self, n_features):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self.output_size]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

            self.m_weights.append(np.zeros((fan_in, fan_out)))
            self.v_weights.append(np.zeros((fan_in, fan_out)))
            self.m_biases.append(np.zeros((1, fan_out)))
            self.v_biases.append(np.zeros((1, fan_out)))

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

    def _backpropagation(self, X_batch, y_batch, activations, zs, t):

        delta = activations[-1] - y_batch

        grad_w_output = np.dot(activations[-2].T, delta) / X_batch.shape[0]
        grad_b_output = np.sum(delta, axis=0) / X_batch.shape[0]

        # apply Adam updates to weights
        self.m_weights[-1] = self.beta1 * self.m_weights[-1] + (1 - self.beta1) * grad_w_output
        self.v_weights[-1] = self.beta2 * self.v_weights[-1] + (1 - self.beta2) * (grad_w_output ** 2)
        m_w_hat = self.m_weights[-1] / (1 - self.beta1**t)
        v_w_hat = self.v_weights[-1] / (1 - self.beta2**t)
        self.weights[-1] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # apply Adam updates to bias
        self.m_biases[-1] = self.beta1 * self.m_biases[-1] + (1 - self.beta1) * grad_b_output
        self.v_biases[-1] = self.beta2 * self.v_biases[-1] + (1 - self.beta2) * (grad_b_output ** 2)
        m_b_hat = self.m_biases[-1] / (1 - self.beta1**t)
        v_b_hat = self.v_biases[-1] / (1 - self.beta2**t)
        self.biases[-1] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # Propagate gradients backward through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(zs[l]) # d_activation(z)
            grad_w_hidden = np.dot(activations[l].T, delta) / X_batch.shape[0]
            grad_b_hidden = np.sum(delta, axis=0) / X_batch.shape[0]

            # apply Adam updates to weights
            self.m_weights[l] = self.beta1 * self.m_weights[l] + (1 - self.beta1) * grad_w_hidden
            self.v_weights[l] = self.beta2 * self.v_weights[l] + (1 - self.beta2) * (grad_w_hidden ** 2)
            m_w_hat = self.m_weights[l] / (1 - self.beta1**t)
            v_w_hat = self.v_weights[l] / (1 - self.beta2**t)
            self.weights[l] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            # apply Adam updates to bias
            self.m_biases[l] = self.beta1 * self.m_biases[l] + (1 - self.beta1) * grad_b_hidden
            self.v_biases[l] = self.beta2 * self.v_biases[l] + (1 - self.beta2) * (grad_b_hidden ** 2)
            m_b_hat = self.m_biases[l] / (1 - self.beta1**t)
            v_b_hat = self.v_biases[l] / (1 - self.beta2**t)
            self.biases[l] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X, y, X_valid, y_valid):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        print("x_train shape: ", X.shape)
        print("y_train shape: ", y.shape)
        print("x_valid shape: ", X_valid.shape)
        print("y_valid shape: ", y_valid.shape)

        # global time step for Adam bias correction
        t = 0

        for epoch in range(self.n_epochs):
            # shuffle datasets
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                t += 1

                activations, zs = self._feedforward(X_batch)
                self._backpropagation(X_batch, y_batch, activations, zs, t)

            activations, _ = self._feedforward(X)
            train_loss = self._binary_cross_entropy(y, activations[-1])
            self.train_loss_history.append(train_loss)
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(y, axis=1)
            self.train_accuracy_history.append(accuracy_score(y_true, y_pred))

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            activations, _ = self._feedforward(X_valid)
            valid_loss = self._binary_cross_entropy(y_valid, activations[-1])
            self.valid_loss_history.append(valid_loss)
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(y_valid, axis=1)
            self.valid_accuracy_history.append(accuracy_score(y_true, y_pred))

            if self._early_stopping():
                break

            print(f"Epoch {epoch+1}/{self.n_epochs}, "
                f"{Fore.YELLOW}Train Loss: {train_loss:.4f}{Style.RESET_ALL}, "
                f"{Fore.CYAN}Valid Loss: {valid_loss:.4f}{Style.RESET_ALL}, "
                f"{Fore.MAGENTA}Precision: {precision:.2f}{Style.RESET_ALL}, "
                f"{Fore.RED}Recall: {recall:.2f}{Style.RESET_ALL}, "
                f"{Fore.GREEN}F1: {f1:.2f}{Style.RESET_ALL}")

        plot_graphs(self.train_loss_history, self.valid_loss_history, self.train_accuracy_history, self.valid_accuracy_history)

    def predict(self, X):
        activations, _ = self._feedforward(X)
        return np.argmax(activations[-1], axis=1)

    def save_model(self, X_train):
        topology = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'input_size': X_train.shape[1],
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'activation': 'relu',
            'activation_output': 'softmax',
        }
        np.savez('mlp_weights.npz', *self.weights, *self.biases)
        with open('mlp_topology.json', 'w') as f:
            json.dump(topology, f)