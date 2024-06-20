import numpy as np
import pandas as pd
from tqdm import tqdm

class FNN():

    class Layer():

        def __init__(self, curr_neurons: int, prev_neurons: int):
            self.W = np.random.randn(prev_neurons, curr_neurons) * np.sqrt(2 / prev_neurons)
            self.B = np.ones((curr_neurons, 1)) * 0.01
            self.neurons = curr_neurons
            self.Z = None
            self.A = None
            self.dZ = None
            self.dW = None
            self.dB = None
            self.D = None
            self.VdW = np.zeros_like(self.W)
            self.VdB = np.zeros_like(self.B)
            self.SdW = np.zeros_like(self.W)
            self.SdB = np.zeros_like(self.B)
    
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, number_hidden_layers: int, sizes: list):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.m = X_train.shape[1]
        self.n = X_train.shape[0]
        self.layers = self.initialize_layers(number_hidden_layers, sizes)
        self.L = len(self.layers)

    def initialize_layers(self, number_hidden_layers: int, sizes: list) -> list:
        layers = []
        input_layer = self.Layer(self.n, 1)
        input_layer.A = self.X_train
        layers.append(input_layer)
        for i in range(0, number_hidden_layers):
            hidden_layer = self.Layer(sizes[i], layers[i].neurons)
            layers.append(hidden_layer)
        output_layer = self.Layer(1, layers[-1].neurons)
        layers.append(output_layer)

        return layers
    
    def activation(self, Z: np.ndarray, derivative = False) -> np.ndarray:
        if self.activation_function == "tanh":
            return self.tanh(Z, derivative)
        elif self.activation_function == "ReLU":
            return self.ReLU(Z, derivative)
        elif self.activation_function == "LeakyReLU":
            return self.LeakyReLU(Z, derivative)
    
    def tanh(self, Z: np.ndarray, derivative: bool) -> np.ndarray:
        if not derivative:
            return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        else:
            return 1 - self.tanh(Z, False) ** 2
    
    def ReLU(self, Z: np.ndarray, derivative: bool) -> np.ndarray:
        if not derivative:
            return np.maximum(0, Z)
        else:
            return Z > 0
    
    def LeakyReLU(self, Z: np.ndarray, derivative: bool) -> np.ndarray:
        if not derivative:
            return np.maximum(0.01 * Z, Z)
        else:
            return np.where(Z > 0, 1, 0.01)
    
    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        #return 1 / (1 + np.exp(-Z))

    def forwardprop(self, on_test_set = False, keep_prob = 1.0) -> np.ndarray:
        for l in range(1, self.L):            
            layer = self.layers[l]
            if on_test_set and l == 1:
                keep_prob = 1
                layer.Z = layer.W.T @ self.X_test + layer.B
            else:
                layer.Z = layer.W.T @ self.layers[l - 1].A + layer.B
            if l == self.L - 1:
                layer.A = self.sigmoid(layer.Z)
                return layer.A
            else:
                layer.A = self.activation(layer.Z, derivative = False)
                layer.D = np.random.rand(layer.A.shape[0], layer.A.shape[1])
                layer.D = (layer.D < keep_prob).astype(int)
                layer.A = layer.A * layer.D
                layer.A = layer.A / keep_prob
    
    def backwardprop(self, Y_hat: np.ndarray, regularization_parameter: float, keep_prob = 1.0) -> None:
        self.layers[-1].dZ = Y_hat - self.Y_train
        for l in reversed(range(1, self.L)):
            layer = self.layers[l]
            layer.dW = (1 / self.m) * (self.layers[l - 1].A @ layer.dZ.T) + (regularization_parameter / self.m) * layer.W
            layer.dB = (1 / self.m) * np.sum(layer.dZ, axis = 1, keepdims = True)
            if l == 1:
                break
            dA = (layer.W @ layer.dZ)
            dA = (layer.W @ layer.dZ) * self.layers[l - 1].D
            dA = dA / keep_prob
            self.layers[l - 1].dZ = dA * self.activation(self.layers[l - 1].Z, derivative = True)
    
    def update_parameters(self, learning_rate: float, optimizer: str, time: int) -> None:
        if optimizer == "Gradient Descent":
            self.gradient_descent(learning_rate = learning_rate)
        elif optimizer == "Adam":
            self.adam(learning_rate = learning_rate, time = time)
    
    def gradient_descent(self, learning_rate: float) -> None:
        for l in range(1, self.L):
            layer = self.layers[l]
            layer.W = layer.W - (learning_rate * layer.dW)
            layer.B = layer.B - (learning_rate * layer.dB)
    
    def adam(self, learning_rate: float, time: int) -> None:
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        for l in range(1, self.L):
            layer = self.layers[l]
            layer.VdW = beta1 * layer.VdW + (1 - beta1) * layer.dW
            layer.VdB = beta1 * layer.VdB + (1 - beta1) * layer.dB
            layer.SdW = beta2 * layer.SdW + (1 - beta2) * (layer.dW ** 2)
            layer.SdB = beta2 * layer.SdB + (1 - beta2) * (layer.dB ** 2)
            VdW_corrected = layer.VdW / (1 - beta1 ** time)
            VdB_corrected = layer.VdB / (1 - beta1 ** time)
            SdW_corrected = layer.SdW / (1 - beta2 ** time)
            SdB_corrected = layer.SdB / (1 - beta2 ** time)

            layer.W = layer.W - learning_rate * (VdW_corrected / (np.sqrt(SdW_corrected) + epsilon))
            layer.B = layer.B - learning_rate * (VdB_corrected / (np.sqrt(SdB_corrected) + epsilon))
    
    def cost(self, Y_hat: np.ndarray):
        Y_hat = np.squeeze(Y_hat)
        Y = np.squeeze(self.Y_train)
        epsilon = 1e-15
        Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
        loss = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / self.m
        return loss
    
    def regularized_cost(self, regularization_parameter):
        weights_sum = 0.0
        for l in range(1, self.L):
            weights_sum += np.sum(np.square(self.layers[l].W))
        return (1 / self.m) * (regularization_parameter / 2) * weights_sum
    
    def fit(self, epochs: int, learning_rate: float, activation_function: str, regularization_parameter: float, keep_prob: float, optimizer = "Gradient Descent", decay_rate = 1.0) -> None:
        self.activation_function = activation_function
        for epoch in range(epochs):
            Y_hat = self.forwardprop(on_test_set = False, keep_prob = keep_prob)
            cost = self.cost(Y_hat)
            L2_regularized_cost = self.regularized_cost(regularization_parameter = regularization_parameter) 
            if epoch % 10 == 0:
                print("Cost : ", cost + L2_regularized_cost)
            self.backwardprop(Y_hat = Y_hat, regularization_parameter = regularization_parameter, keep_prob = keep_prob)
            self.update_parameters(learning_rate = learning_rate, optimizer = optimizer, time = epoch + 1)
            learning_rate = (1 / (1 + decay_rate * epoch)) * learning_rate
    
    def test(self, on_test_set = False):
        Y_hat = self.forwardprop(on_test_set = on_test_set)
        Y_hat = np.squeeze(Y_hat)
        if on_test_set:
            Y = np.squeeze(self.Y_test)
        else:    
            Y = np.squeeze(self.Y_train)
        tp = 0.001
        fp = 0.001
        tn = 0.001
        fn = 0.001
        for i in range(len(Y_hat)):
            y_hat = 1 if Y_hat[i] >= 0.5 else 0
            y = Y[i]
            if y_hat == 1 and y == 1:
                tp += 1
            elif y_hat == 1 and y == 0:
                fp += 1
            elif y_hat == 0 and y == 0:
                tn += 1
            else:
                fn += 1
    
        total = tp + fp + tn + fn
        classification_accuracy = ((tp + tn) / total) * 100
        misclassification_rate = ((fp + fn) / total) * 100
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * recall * precision / (recall + precision)
        print("------Performance------")
        print("Classification Accuracy: ", classification_accuracy, "%")
        print("Misclassification Rate: ", misclassification_rate, "%")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-Score: ", f1score)

if __name__ == "__main__":
    test = pd.read_csv('/home/vaibhav/Desktop/AI/Heart Disease Prediction/test.csv')
    train = pd.read_csv('/home/vaibhav/Desktop/AI/Heart Disease Prediction/train.csv')
    X_train = train.iloc[:, :-1].values.T
    Y_train = train.iloc[:, -1:].values.T
    X_test = test.iloc[:, :-1].values.T
    Y_test = test.iloc[:, -1:].values.T
    model = FNN(X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, number_hidden_layers = 3, sizes = [100, 100, 64])
    model.fit(epochs = 500, learning_rate = 0.2, activation_function = "ReLU", regularization_parameter = 0, keep_prob = 1.0, optimizer = "Adam", decay_rate = 0.0)
    model.test(on_test_set = False)
    model.test(on_test_set = True)