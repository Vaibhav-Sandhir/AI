import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dataset(size =  200, timesteps = 25):
    x, y = [], []
    sin_wave = np.sin(np.arange(size))
    for step in range(sin_wave.shape[0]-timesteps):
        x.append(sin_wave[step:step+timesteps])
        y.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)

class RNN:
    def __init__(self, x, y, hidden_units):
        self.x = x
        self.y = y
        self.hidden_units = hidden_units
        self.Wx = np.random.randn(self.hidden_units, self.x.shape[2]) / np.sqrt(self.x.shape[2])
        self.Wa = np.random.randn(self.hidden_units, self.hidden_units) / np.sqrt(self.hidden_units)
        self.Wy = np.random.randn(self.hidden_units, self.y.shape[1]) / np.sqrt(self.hidden_units)
        self.B = np.zeros((self.hidden_units, self.x.shape[2]))
        self.C = np.zeros((self.y.shape[1], self.y.shape[1]))
    
    def forwardprop(self, sample):
        x_sample = self.x[sample]
        y_sample = self.y[sample]
        self.activations = []
        self.inputs = []
        at = np.zeros((self.hidden_units, x_sample.shape[1]))
        self.activations.append(at)

        for step in range(len(x_sample)):
            xt = x_sample[step]
            xt = xt.reshape(1, 1)
            zt = self.Wa @ self.activations[step] + self.Wx @ xt + self.B
            at = np.tanh(zt)
            ot = self.Wy.T @ at + self.C
            yt_hat = ot
            self.activations.append(at)
            self.inputs.append(xt)

        self.error = yt_hat - y_sample   
        self.loss = 0.5*self.error**2
        self.yt_hat = yt_hat

    def backprop(self):
        n = len(self.inputs)
        dC = self.error
        dB = np.zeros(self.B.shape)
        dWy = (self.error @ self.activations[-1].T).T
        dWx = np.zeros(self.Wx.shape)
        dWa = np.zeros(self.Wa.shape)
        dat = self.error @ self.Wy.T 
        
        for step in reversed(range(n)):
            dzt = (1 - self.activations[step] ** 2) * dat.T
            dB += dzt
            dWx += dzt @ self.inputs[step]
            if step > 0:
                dWa += dzt @ self.activations[step - 1].T  
            dat = dzt.T @ self.Wa
 
        dWy = np.clip(dWy, -1, 1)
        dWx = np.clip(dWx, -1, 1)
        dWa = np.clip(dWa, -1, 1)
        dB = np.clip(dB, -1, 1)
        dC = np.clip(dC, -1, 1)

        self.Wy -= self.lr * dWy
        self.Wx -= self.lr * dWx
        self.Wa -= self.lr * dWa
        self.B -= self.lr * dB
        self.C -= self.lr * dC 

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
    rnn = RNN(x,y,100)
    rnn.train(50,0.01)
    rnn.test(x_test, y_test)
    plt.tight_layout()
    plt.subplot(122)
    plt.plot([i for i in range(len(x_test))],y_test,np.array(rnn.outputs).reshape(y_test.shape))
    plt.savefig("Prediction from RNN")
    plt.show()