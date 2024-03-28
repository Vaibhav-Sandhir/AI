import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, curr_neurons, prev_neurons):
        rows = curr_neurons
        cols = prev_neurons
        self.W = np.random.randn(rows, cols) * np.sqrt(2 / cols)
        self.B = np.ones((rows, 1)) * 0.01
        self.A = None
        self.Z = None
        self.dW = None
        self.dB = None

class Model:
    def __init__(self, X_train, Y_train, alpha):
        self.layers = []
        self.X_train = X_train
        self.Y_train = Y_train
        self.Y1 = None
        self.alpha = alpha

    def addLayer(self, curr_neurons, prev_neurons):
        layer = Layer(curr_neurons, prev_neurons)
        if len(self.layers) == 0:
            layer.A = self.X_train
        self.layers.append(layer)

    def ReLU(self, Z):
        A = np.maximum(0, Z)
        return A
    
    def ReLU_derivative(self, Z):
        return Z > 0
    
    def ELU(self, Z, alpha=1.0):
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))
    
    def ELU_derivative(self, Z, alpha=1.0):
        return np.where(Z > 0, 1, alpha * np.exp(Z))
    
    def SELU(self, Z, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
        return scale * np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

    def SELU_derivative(self, Z, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
        return scale * np.where(Z > 0, 1, alpha * np.exp(Z))
    
    def leakyReLU(self, Z):
        A = np.maximum(0.01 * Z, Z)
        return A
    
    def leakyReLU_derivative(self, Z):
        dZ = np.where(Z > 0, 1, 0.01)
        return dZ
    
    def forwardProp(self):
        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            curr_layer.Z = (curr_layer.W @ prev_layer.A) + curr_layer.B
            if i == len(self.layers) - 1:
                curr_layer.A = curr_layer.Z
                self.Y1 = curr_layer.A
            else:
                curr_layer.A = self.SELU(curr_layer.Z)

    def backProp(self):
        m = self.X_train.shape[1]
        l = len(self.layers) - 1
        error = (self.Y1 - self.Y_train) / self.Y_train
        while True:
            curr_layer = self.layers[l]
            prev_layer = self.layers[l - 1]
            curr_layer.dW = (1 / m) * (error @ prev_layer.A.T)
            curr_layer.dB = (1 / m) * np.sum(error)
            if l == 1:
                break
            error = (curr_layer.W.T @ error) * self.SELU_derivative(prev_layer.Z)
            l = l - 1

    def updateParameters(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.W = layer.W - (self.alpha * layer.dW)
            layer.B = layer.B - (self.alpha * layer.dB)

    def getAccuracy(self):
        error = np.abs(self.Y_train - self.Y1)
        m = self.X_train.shape[1]
        for i in range(0, m):
            error[0, i] = error[0, i] / self.Y_train[0, i]
            error[0, i] = 1 - error[0, i]
        accuracy = np.sum(error) / m
        return accuracy


    def run(self, epochs):
        for i in range(epochs):
            self.forwardProp()
            self.backProp()
            self.updateParameters()
            if i % 1000 == 0:
                print("Accuracy: ", self.getAccuracy()) 

    def predict(self, X_test, Y_test):
        self.layers[0].A = X_test
        self.forwardProp()
        m = self.Y1.shape[1]
        results = {"Actual Price":list(Y_test[0]), "Predicted Price":list(self.Y1[0])}
        df = pd.DataFrame(results)
        acc = []
        error = 0
        for i in range(0, m):
            error += abs(self.Y1[0,i] - Y_test[0,i]) / Y_test[0, i]
            acc.append(1 - (abs(self.Y1[0,i] - Y_test[0,i]) / Y_test[0, i]))
        error = error / m
        df['Accuracy'] = acc
        print("Accuracy: ", 1 - error) 
        df.to_csv('/home/vaibhav/Desktop/AI/Laptop Price Prediction/Results.csv')
        plt.plot(df.index, df['Accuracy'], marker='o', linestyle='', color='red', markersize=5)
        plt.title('Accuracy for Individual Estimations')
        plt.xlabel('Index')
        plt.ylabel('Accuracy')
        plt.savefig('/home/vaibhav/Desktop/AI/Laptop Price Prediction/Accuracy.png')                             

def getData():
    data = pd.read_csv('/home/vaibhav/Desktop/AI/Laptop Price Prediction/preprocessed_data.csv')
    data.drop(data.columns[0], axis = 1, inplace=True)
    m, n = data.shape
    Y = data['Final Price']
    Y = np.array(Y)
    data.drop('Final Price', axis = 1, inplace = True)
    X = np.array(data)
    m = int(0.9 * m)
    X_train = X[:m]
    X_test = X[m:]
    Y_train = Y[:m]
    Y_test = Y[m:]
    return X_train.T, Y_train.reshape(1, -1), X_test.T, Y_test.reshape(1, -1)

def main():
    X_train, Y_train, X_test, Y_test = getData()
    model = Model(X_train, Y_train, 0.01)
    model.addLayer(209, 1)
    model.addLayer(16, 209)
    model.addLayer(16, 16)
    model.addLayer(1, 16)
    model.run(10000)
    model.predict(X_test, Y_test)    

if __name__ == '__main__':
    main()	