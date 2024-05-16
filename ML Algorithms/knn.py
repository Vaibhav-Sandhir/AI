import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]



def main():
    iris = datasets.load_iris()
    X, Y = iris.data, iris.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    plt.figure()
    plt.scatter(X[:,2], X[:,3], c = Y)
    plt.show()

    classifier = KNN(5, X_train, Y_train)
    predictions = classifier.predict(X_test)

    acc = np.sum(predictions == Y_test) / len(Y_test)
    print(acc)

if __name__ == "__main__":
    main()


