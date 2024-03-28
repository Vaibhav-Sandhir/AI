import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Layer:
	def __init__(self, curr_neurons, prev_neurons):
		rows = curr_neurons
		cols = prev_neurons
		self.W = np.random.rand(rows, cols) - 0.5
		self.B = np.random.rand(rows, 1) - 0.5
		self.A = None
		self.Z = None
		self.dW = None
		self.dB = None

class Model:
	def __init__(self, X, Y, alpha):
		self.layers = []
		self.X = X
		self.Y = Y
		self.Y1 = None
		self.alpha = alpha
		one_hot = np.zeros((10, Y.size))
		for i,num in enumerate(Y):
			one_hot[num, i] = 1
		self.one_hot_Y = one_hot
	
	def addLayer(self, curr_neurons, prev_neurons):
		layer = Layer(curr_neurons, prev_neurons)
		if len(self.layers) == 0:
			layer.A = self.X
		self.layers.append(layer)	
	
	def ReLU(self, Z):
		A = np.maximum(0, Z)
		return A

	def softmax(self, Z):
		A = np.exp(Z) / sum(np.exp(Z))
		return A		
	
	def derivative_ReLU(self, Z):
		return Z > 0
	
	def getPredictions(self):
		return np.argmax(self.Y1, 0)
	
	def getAccuracy(self, predictions):
		return np.sum(predictions == self.Y) / self.Y.size			
				
	def forwardProp(self):
		for i in range(1, len(self.layers)):
			curr_layer = self.layers[i]
			prev_layer = self.layers[i - 1]
			curr_layer.Z = (curr_layer.W @ prev_layer.A) + curr_layer.B
			if i == len(self.layers) - 1:
				curr_layer.A = self.softmax(curr_layer.Z)
				self.Y1 = curr_layer.A
			else:
				curr_layer.A = self.ReLU(curr_layer.Z)	
				
	def backProp(self):
		m = self.X.shape[1]
		l = len(self.layers) - 1
		error = self.Y1 - self.one_hot_Y
		while True:
			self.layers[l].dW = (1 / m) * error @ self.layers[l - 1].A.T
			self.layers[l].dB = (1 / m) * np.sum(error)
			if l == 1:
				break
			error = (self.layers[l].W.T @ error) * self.derivative_ReLU(self.layers[l - 1].Z)
			l = l - 1
	
	def updateParameters(self):
		for i in range(1, len(self.layers)):
			layer = self.layers[i]
			layer.W = layer.W - self.alpha * layer.dW
			layer.B = layer.B - self.alpha * layer.dB
	
	def run(self, epochs):
		for i in range(epochs):
			self.forwardProp()
			self.backProp()
			self.updateParameters()
			if i % 10 == 0:
				print("Accuracy: ", self.getAccuracy(self.getPredictions()))

	def predict(self, X_test):
		self.layers[0].A = X_test
		m = X_test.shape[1]
		self.forwardProp()
		self.Y1 = np.argmax(self.Y1, 0)
		path = '/home/vaibhav/Desktop/LLM/Perceptron/predictions.csv'
		np.savetxt(path, self.Y1, delimiter=',')


																		
						
def getData():
	data = pd.read_csv('/home/vaibhav/Desktop/LLM/Perceptron/train.csv')
	data_t = pd.read_csv('/home/vaibhav/Desktop/LLM/Perceptron/test.csv')
	data = np.array(data)
	X_test = np.array(data_t)
	data = data.T
	X_test = X_test.T
	Y = data[0]
	X = data[1:785]
	X = X / 255.
	X_test = X_test / 255.
	return X, Y, X_test

def main():
	X, Y, X_test = getData()
	model = Model(X, Y, 0.1)
	model.addLayer(784, 1)
	model.addLayer(16, 784)
	model.addLayer(16, 16)
	model.addLayer(10, 16)
	model.run(2000)
	model.predict(X_test)

if __name__ == '__main__':
	main()	
	
