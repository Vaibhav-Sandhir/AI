import numpy as np

def ReLU(Z, derivative = False):
    if not derivative:
        return np.maximum(0, Z)
    else:
        return Z > 0

def softmax(Z):
	A = np.exp(Z) / sum(np.exp(Z))
	return A	

def convolve_single_step(a_slice, w, b):
    s = np.sum(a_slice * w)
    z = s + b.item()
    return z

def convolve_forward(A_prev, layer):
    W = layer.W
    B = layer.B
    stride = layer.stride

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, _, n_C_prev, n_C = W.shape
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))

    for i in range(0, m):
        a_prev = A_prev[i]
        for h in range(0, n_H):
            vert_start = stride * h
            vert_end = stride * h + f
            for w in range(0, n_W):
                horiz_start = stride * w
                horiz_end = stride * w + f
                for c in range(0, n_C):
                    a_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]
                    w_c = W[:, :, :, c]
                    b_c = B[:, :, :, c]
                    Z[i, h, w, c] = convolve_single_step(a_slice, w_c, b_c)

    layer.Z = Z
    return ReLU(Z)

def pool_forward(A_prev, layer):
    P = layer.P
    stride = layer.stride
    mode = layer.mode

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, _, n_C = P.shape
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    A_p = np.zeros((m, n_H, n_W, n_C))
    for i in range(0, m):
        a_prev = A_prev[i]
        for h in range(0, n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            for w in range(0, n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                for c in range(0, n_C):
                    a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A_p[i, h, w, c] = np.max(a_slice_prev)
                    else:
                        A_p[i, h, w, c] = np.mean(a_slice_prev)
    return A_p

def linear_forward(A_prev, layer, output = False):
    layer.Z = layer.W.T @ A_prev + layer.B
    if output:
        return softmax(layer.Z)
    else:
        return ReLU(layer.Z)

def create_mask(X):
    mask = (X == np.max(X))
    return mask

def distribute_value(dZ, shape):
    average = np.prod(shape)
    a = (dZ / average)*np.ones(shape)
    return a

def convolve_backward(prev_layer, layer):
    layer.dZ = layer.dA * ReLU(layer.Z, derivative = True)
    if layer.input:
        A_prev = X_train
    else: 
        A_prev = prev_layer.A
    W = layer.W
    B = layer.B
    stride = layer.stride

    m, n_H, n_W, n_C = layer.dZ.shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, _, n_C_prev, n_C = W.shape
    dA_prev = np.zeros_like(A_prev)

    for i in range(0, m):
        a_prev = A_prev[i]
        da_prev = dA_prev[i]
        for h in range(0, n_H):
            vert_start = stride * h
            vert_end = stride * h + f
            for w in range(0, n_W):
                horiz_start = stride * w
                horiz_end = stride * w + f
                for c in range(0, n_C):
                    a_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * layer.dZ[i, h, w, c]
                    layer.dW[:, :, :, c] += a_slice * layer.dZ[i, h, w, c]
                    layer.dB[:, :, :, c] += layer.dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev
    
    if not layer.input:
        prev_layer.dA = dA_prev

def pool_backward(prev_layer, layer):
    stride = layer.stride
    A_prev = prev_layer.A
    f, _, n_C = layer.P.shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = layer.dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m): 
        a_prev = A_prev[i,:,:,:]
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    vert_start  = h * stride
                    vert_end    = h * stride + f
                    horiz_start = w * stride
                    horiz_end   = w * stride + f
                    if layer.mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * layer.dA[i, h, w, c]
                    elif layer.mode == "average":
                        da = layer.dA[i, h, w, c]
                        shape = (f,f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
    prev_layer.dA = dA_prev

def linear_backward(A_prev, layer, Y_train, m):
    if layer.output:
        layer.dZ = layer.A - Y_train
    else:
        layer.dZ = layer.dA * ReLU(layer.Z, derivative = True)
    if(len(A_prev.shape) == 4):
        m, s1, s2, s3 = A_prev.shape
        A_prev = A_prev.reshape(s1*s2*s3, m)
    layer.dW = (1 / m) * (A_prev @ layer.dZ.T)
    layer.dB = (1 / m) * np.sum(layer.dZ, axis = 1, keepdims = True)
    dA = layer.W @ layer.dZ
    return dA

class ConvLayer:
    
    def __init__(self, f, n_C_prev, n_C, stride, iput = "False"):
        self.W = np.zeros((f, f, n_C_prev, n_C))
        self.B = np.ones((1, 1, 1, n_C)) * 0.01
        self.Z = None
        self.A = None
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        self.dZ = None
        self.dA = None
        self.stride = stride
        self.name = "Convolution"
        self.input = iput

class PoolLayer:

    def __init__(self, f, n_C, stride, mode = "max"):
        self.P = np.zeros((f, f, n_C))
        self.stride = stride
        self.A = None
        self.dA = None
        self.mode = mode
        self.name = "Pooling"

class LinearLayer:

    def __init__(self, curr_neurons, prev_neurons, transition = False, output = False):
        self.W = np.random.randn(prev_neurons, curr_neurons) * np.sqrt(2 / prev_neurons)
        self.B = np.ones((curr_neurons, 1)) * 0.01
        self.neurons = curr_neurons
        self.Z = None
        self.A = None
        self.dA = None
        self.dZ = None
        self.dW = None
        self.dB = None
        self.transition = transition
        self.output = output
        self.name = "Linear"

class CNN:

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.layers = []
        self.m, _, _, _ = X_train.shape 
    
    def addConvLayer(self, f, n_C_prev, n_C, stride, iput = False):
        layer = ConvLayer(f, n_C_prev, n_C, stride, iput)
        self.layers.append(layer)
    
    def addPoolLayer(self, f, n_C, stride, mode = "max"):
        layer = PoolLayer(f, n_C, stride, mode)
        self.layers.append(layer)
    
    def addLinearLayer(self, curr_neurons, prev_neurons, transition = False, output = False):
        layer = LinearLayer(curr_neurons, prev_neurons, transition, output)
        self.layers.append(layer)
    
    def CELoss(self, Y_hat):
        m = self.Y_train.shape[0]
        epsilon = 1e-15
        Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
        cross_entropy = -np.sum(self.Y_train * np.log(Y_hat)) / m

        return cross_entropy
    
    def forward_pass(self):
        for l in range(0, self.L):
            layer = self.layers[l]
            
            if l == 0:
                A_prev = self.X_train
            else:
                A_prev = self.layers[l - 1].A

            if layer.name == "Convolution":
                layer.A = convolve_forward(A_prev, layer)  
            elif layer.name == "Pooling":
                layer.A = pool_forward(A_prev, layer)
            elif layer.name == "Linear":
                if layer.transition:
                    m, s1, s2, s3 = A_prev.shape
                    A_prev = A_prev.reshape(s1 * s2 * s3, m)
                layer.A = linear_forward(A_prev, layer)
            
            if l == self.L - 1:
                return layer.A
    
    def backward_pass(self):
        for l in reversed(range(0, self.L)):
            layer = self.layers[l]
            
            if l == 0:
                prev_layer = None
            else:
                prev_layer = self.layers[l - 1]
            
            if layer.name == "Convolution":
                convolve_backward(prev_layer, layer)    
            elif layer.name == "Pooling":
                pool_backward(prev_layer, layer)
            elif layer.name == "Linear":
                prev_layer.dA = linear_backward(prev_layer.A, layer, self.Y_train, self.m)
                if layer.transition:
                    A_prev = prev_layer.A
                    m, s1, s2, s3 = A_prev.shape
                    prev_layer.dA = prev_layer.dA.reshape(m, s1, s2, s3)

    def update_parameters(self, learning_rate):
        for l in range(0, self.L):
            layer = self.layers[l]
            if layer.name == "Convolution" or layer.name == "Linear":
                layer.W = layer.W - learning_rate * layer.dW
                layer.B = layer.B - learning_rate * layer.dB
    
    def train(self, epochs, learning_rate):
        self.L = len(self.layers)
        for epoch in range(epochs):
            Y_hat = self.forward_pass()
            loss = self.CELoss(Y_hat)
            print("Loss: ", loss)
            self.backward_pass()
            self.update_parameters(learning_rate)


def load_data():
    X_train = np.load('x_train.npy')
    Y_train = np.load('y_train.npy')
    X_test = np.load('x_test.npy')
    Y_test = np.load('y_test.npy')

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    # print(X_train.shape)
    # print(Y_train.shape)
    X_train = X_train[:100, :, :, :]
    Y_train = Y_train[:100, :].T
    # print(X_train.shape)
    # print(Y_train.shape)
    cnn = CNN(X_train, Y_train, X_test, Y_test)
    cnn.addConvLayer(f = 7, n_C_prev = 3, n_C = 16, stride = 2, iput = True)
    cnn.addPoolLayer(f = 3, n_C = 16, stride = 2, mode = "max")
    cnn.addConvLayer(f = 3, n_C_prev = 16, n_C = 32, stride = 2)
    cnn.addPoolLayer(f = 3, n_C = 32, stride = 2, mode = "max")
    cnn.addLinearLayer(curr_neurons = 64, prev_neurons = 800, transition = True)
    cnn.addLinearLayer(curr_neurons = 16, prev_neurons = 64)
    cnn.addLinearLayer(curr_neurons = 6, prev_neurons = 16, output = True)
    cnn.train(epochs = 100, learning_rate = 0.01)