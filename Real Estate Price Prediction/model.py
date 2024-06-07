import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultiplePolynomialRegression:

    def __init__(self, train_file, test_file):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        self.X_train = train.iloc[:, :-1].values
        self.X_test = test.iloc[:, :-1].values
        self.Y_train = train.iloc[:, -1:].values
        self.Y_test = test.iloc[:, -1:].values
        self.m = len(self.X_train)
        self.n = len(self.X_train[0])
        self.W = np.zeros((1, self.n))
        self.B = np.zeros((1, 1))
        self.transform()

    def transform(self):
        for i in range(0, self.m):
            #self.X_train[i][0] = np.sin(self.X_train[i][0])
            self.X_train[i][0] = self.X_train[i][0] ** 3
            self.X_train[i][1] = self.X_train[i][1] ** 3
            #self.X_train[i][2] = 1 / self.X_train[i][2]
            self.X_train[i][3] = self.X_train[i][3] ** 3
            self.X_train[i][4] = self.X_train[i][4] ** 4
            self.X_train[i][5] = self.X_train[i][5] ** 3
        for i in range(0, len(self.X_test)):
            #self.X_test[i][0] = np.sin(self.X_test[i][0])
            self.X_test[i][0] = self.X_test[i][0] ** 3
            self.X_test[i][1] = self.X_test[i][1] ** 3
            #self.X_test[i][2] = 1 / self.X_test[i][2]
            self.X_test[i][3] = self.X_test[i][3] ** 3
            self.X_test[i][4] = self.X_test[i][4] ** 4
            self.X_test[i][5] = self.X_test[i][5] ** 3
    
    def predict(self, xi):
        y_hat = self.W @ xi.T + self.B
        return y_hat
    
    def loss(self, y_hat, y):
        loss = (y_hat - y)** 2  * 1 / 2*self.m
        return loss
    
    def partial_derivatives(self, y_hat, y, xi):
        dW = (y_hat - y) * xi * 1 / self.m
        dB = (y_hat - y) * 1 / self.m
        return dW, dB
    
    def update_parametrs(self, dW, dB, alpha):
        self.W = self.W - alpha * dW
        self.B = self.B - alpha * dB
    
    def train(self, alpha, epochs):
        losses = []
        iterations = []
        for epoch in range(0, epochs):
            loss = 0.0
            dW = 0.0
            dB = 0.0
            for i in range(0, self.m):
                xi = self.X_train[i].reshape(1, self.n)
                y = self.Y_train[i].reshape(1, 1)
                y_hat = self.predict(xi)
                loss += self.loss(y_hat, y)
                p1, p2 = self.partial_derivatives(y_hat, y, xi)
                dW += p1
                dB += p2
            self.update_parametrs(dW, dB, alpha)
            losses.append(loss.reshape(1,))
            iterations.append(epoch)
            print("Epoch : ", epoch, " Loss : ", loss)
        plt.figure()
        plt.plot(iterations, losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig('Loss vs Epoch')


    
    def test(self):
        l = len(self.Y_test)
        mae = 0
        predictions = []
        targets = []
        for i in range(0, len(self.X_test)):
            xi = self.X_test[i].reshape(1, self.n)
            y = self.Y_test[i].reshape(1, 1)
            y_hat = self.predict(xi)
            mae += abs((y_hat - y)) / l
            predictions.append(y_hat.reshape(1, ))
            targets.append(y.reshape(1, ))
        print("---------------")
        print(mae)
        plt.figure()
        plt.scatter(predictions, targets)
        plt.plot([min(predictions), max(predictions)], [min(targets), max(targets)], color='red')
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.savefig("Predicted vs Target")

if __name__ == "__main__":
    model = MultiplePolynomialRegression('train.csv', 'test.csv')
    model.train(0.01, 300)
    model.test()