import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dataset(size = 200, timesteps = 25):
    x, y = [], []
    sin_wave = np.sin(np.arange(size))
    for step in range(sin_wave.shape[0] - timesteps):
        x.append(sin_wave[step:step + timesteps])
        y.append(sin_wave[step + timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)

class Layer:
    def __init__(self, x, y, hidden_units, num, l):
        if num == 0:
            self.Wx = np.random.randn(hidden_units, x.shape[2]) / np.sqrt(x.shape[2])
        else:
            self.Wx = np.random.randn(hidden_units, hidden_units) / np.sqrt(hidden_units)               
        self.Wh = np.random.randn(hidden_units, hidden_units) / np.sqrt(hidden_units)
        self.Wy = np.random.randn(hidden_units, y.shape[1]) / np.sqrt(hidden_units)
        self.B = np.zeros((hidden_units, x.shape[2]))
        self.C = np.zeros((y.shape[1], y.shape[1]))
        self.dWx = np.zeros(self.Wx.shape) 
        self.dWh = np.zeros(self.Wh.shape)
        self.dB = np.zeros(self.B.shape)
        self.dht = None
        self.acivations = []
        self.inputs = []   


class RNN:
    def __init__(self, x, y, hidden_units, l):
        self.x = x
        self.y = y
        self.hidden_units = hidden_units
        self.l = l
        self.layers = []
        for i in range(0, l):
            self.layers.append(Layer(x, y, hidden_units, i, l))

    def forwardprop(self, sample):
        x_sample = self.x[sample]
        y_sample = self.y[sample]
        at = np.zeros((self.hidden_units, x_sample.shape[1]))
        for layer in self.layers:
            layer.activations = []
            layer.activations.append(at)
            layer.inputs = []
        for xt in x_sample:
            self.layers[0].inputs.append(xt.reshape(1, 1))  
        yt_hat = None

        for step in range(len(x_sample)):
            for l in range(0, self.l):
                layer = self.layers[l]
                inp = layer.inputs[step]  
                zt = layer.Wh @ layer.activations[step] + layer.Wx @ inp + layer.B
                at = np.tanh(zt)
                layer.activations.append(at)    
                if l == self.l - 1: 
                    ot = layer.Wy.T @ at + layer.C
                    yt_hat = ot
                else:    
                    self.layers[l + 1].inputs.append(at)
        
        self.error = yt_hat - y_sample
        self.loss = 0.5*self.error**2
        self.yt_hat = yt_hat

    def backprop(self):
        l_layer = self.layers[self.l - 1]
        n = len(l_layer.inputs)
        l_layer.dC = self.error
        l_layer.dWy = (self.error @ l_layer.activations[-1].T).T
        l_layer.dht = self.error @ l_layer.Wy.T 

        for step in reversed(range(n)):
            for l in reversed(range(self.l)):
                layer = self.layers[l]
                dzt = (1 - layer.activations[step] ** 2) * layer.dht.T
                layer.dB += dzt
                layer.dWx += dzt @ layer.inputs[step].T
                if step > 0:
                    layer.dWh += dzt @ layer.activations[step - 1].T

                if l > 0:
                    p_layer = self.layers[l - 1]
                    p_layer.dht = (p_layer.Wh @ dzt).T

                layer.dht = dzt.T @ layer.Wh

        for layer in self.layers:
            layer.dWx = np.clip(layer.dWx, -1, 1)
            layer.dWh = np.clip(layer.dWh, -1, 1)
            layer.dB = np.clip(layer.dB, -1, 1)
            layer.Wx -= self.lr * layer.dWx
            layer.Wh -= self.lr * layer.dWh
            layer.B -= self.lr * layer.dB
        l_layer.dC = np.clip(l_layer.dC, -1, 1)
        l_layer.dWy = np.clip(l_layer.dWy, -1, 1)
        l_layer.Wy -= self.lr * l_layer.dWy
        l_layer.C -= self.lr * l_layer.dC
    
    def train(self, epochs, learning_rate):
        self.lr = learning_rate
        for epoch in tqdm(range(epochs)):
            for sample in range(self.x.shape[0]):
                self.forwardprop(sample)
                self.backprop()

    def test(self,x,y):
        self.x = x
        self.y = y
        self.outputs = []
        for sample in range(len(x)):
            self.forwardprop(sample)
            self.outputs.append(self.yt_hat)


if __name__ == "__main__":
    x,y = dataset()
    x_test, y_test = dataset(300)
    x_test = x_test[250:]
    y_test = y_test[250:]
    rnn = RNN(x,y,100, 2)
    rnn.train(60,0.001)
    rnn.test(x_test, y_test)
    plt.tight_layout()
    plt.subplot(122)
    plt.plot([i for i in range(len(x_test))],y_test,np.array(rnn.outputs).reshape(y_test.shape))
    plt.savefig("Prediction from DRNN")
    plt.show()
    
                    





                    


            




