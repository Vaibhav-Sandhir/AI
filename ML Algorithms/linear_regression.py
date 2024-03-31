import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.W = np.zeros(X.shape[1])
        self.B = 0

    def run(self, alpha, epochs):
        m = self.X.shape[1]
        for i in range(0, epochs):
            Y1 = np.dot(self.X, self.W) + self.B
            error = Y1 - self.Y
            dW = (1 / m) * np.dot(error, self.W)
            dB = (1 / m) * error
            self.W = self.W - alpha * dW
            self.B = self.B - alpha * dB
        return Y1    
        


def getData():
    X = np.random.randn(1, 500)
    Y = np.random.randint(1, 5) * X + np.random.randint(1, 5)
    return X, Y

def main():
    X, Y = getData()
    model = LinearRegression(X, Y)
    Y1 = model.run(0.01, 1000000)
    plt.scatter(X.flatten(), Y.flatten(), label='Actual')
    plt.plot(X.flatten(), Y1.flatten(), color='red', label='Predicted')
    plt.savefig('Prediction.png')

if __name__ == '__main__':
    main()

