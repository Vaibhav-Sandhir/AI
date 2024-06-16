import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():

    def __init__(self, train_file, test_file):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        self.X_train = train.iloc[:, :-1].values
        self.Y_train = train.iloc[:, -1:].values
        self.X_test = test.iloc[:, :-1].values
        self.Y_test = test.iloc[:, -1:].values
        self.m, self.n = self.X_train.shape
        self.W = np.ones((1, self.n)) * 0.1
        self.B = np.ones((1, 1)) * 0.1             

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, xi, absolute = False):
        z = self.W @ xi.T + self.B
        y_hat = self.sigmoid(z)
        if absolute:
            return 1 if y_hat > 0.5 else 0
        return y_hat
    
    def loss(self, y_hat, y):
        loss = -1 / self.m * (y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat))
        return loss
    
    def partial_derivatives(self, y_hat, y, xi):
        dW = 1 / self.m * (y_hat - y) @ xi
        dB = 1 / self.m * (y_hat - y)
        return dW, dB
    
    def update_parameters(self, alpha, dW, dB):
        self.W = self.W - alpha * dW
        self.B = self.B - alpha * dB
    
    def train(self, alpha, epochs):
        losses = []
        iterations = []
        for epoch in range(0, epochs):
            loss = 0.0
            dW = np.zeros((1, self.n))
            dB = np.zeros((1, 1))
            for i in range(0, self.m):
                xi = self.X_train[i].reshape(1, self.n)
                y = self.Y_train[i].reshape(1, 1)
                y_hat = self.predict(xi)
                loss += self.loss(y_hat, y)
                p1, p2 = self.partial_derivatives(y_hat, y, xi)
                dW += p1
                dB += p2
            losses.append(loss.reshape(1, ))
            iterations.append(epoch)
            self.update_parameters(alpha, dW, dB)
            if epoch % 100 == 0:
                print("Loss : ", loss)
        plt.figure()
        plt.plot(iterations, losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("Loss vs Epochs")
    
    def test(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        predictions = []
        for i in range(0, len(self.X_test)):
            xi = self.X_test[i].reshape(1, self.n)
            y = self.Y_test[i].reshape(1, 1)
            y_hat = self.predict(xi, absolute = True)
            predictions.append(y_hat)
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
        plt.figure()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("Confusion Matrix")

if __name__ == "__main__":
    model = LogisticRegression("/home/vaibhav/Desktop/AI/Heart Disease Prediction/train.csv", "/home/vaibhav/Desktop/AI/Heart Disease Prediction/test.csv")
    model.train(0.001, 800)
    model.test()