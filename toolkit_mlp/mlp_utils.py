import numpy as np

class MLP:
    def __init__(self, hidden_layer_sizes=(30, 30), learning_rate=0.01, n_epochs=1000, batch_size=32):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.weights_history = []
        self.biases_history = []
        self.loss_history = []

    def _relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def _initialize_parameters(self, n_features):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [2]
        # print("Layer sizes:", layer_sizes)
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def _forward_pass(self, X):
        activations = [X]
        # print("Input shape:", X.shape)
        zs = []
        for i in range(len(self.weights) - 1):
            print("self.weights[i].shape:", self.weights[i].shape)
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self._relu(z)
            activations.append(a)

        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z_output)
        y_pred = self._relu(z_output)
        activations.append(y_pred)

        return activations, zs
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        self._initialize_parameters(n_features)
        # self.weights_history.append([w.copy() for w in self.weights])
        # self.biases_history.append([b.copy() for b in self.biases])
        # activations, _ = self._forward_pass(X)
        # initial_loss = self._compute_loss(y, activations[-1])
        # self.loss_history.append(initial_loss)

        for epoch in range(self.n_epochs):
            # shuffle datasets
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                activations, zs = self._forward_pass(X_batch)
                y_pred = activations[-1]

                delta = y_pred - y_batch
                dW = np.dot(activations[-2].T, delta) / X_batch.shape[0]
                db = np.sum(delta, axis=0) / X_batch.shape[0]
                self.weights[-1] -= self.learning_rate * dW
                self.biases[-1] -= self.learning_rate * db

                for l in range(len(self.weights) - 2, -1, -1):
                    delta = np.dot(delta, self.weights[l+1].T) * self._relu_derivative(zs[l]) # d_activation(z)
                    dW = np.dot(activations[l].T, delta) / X_batch.shape[0]
                    db = np.sum(delta, axis=0) / X_batch.shape[0]

                    self.weights[l] -= self.learning_rate * dW
                    self.biases[l] -= self.learning_rate * db

            self.weights_history.append([w.copy() for w in self.weights])
            self.biases_history.append([b.copy() for b in self.biases])

            activations, _ = self._forward_pass(X)
            epoch_loss = self._compute_loss(y, activations[-1])
            self.loss_history.append(epoch_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")
        return self