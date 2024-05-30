import numpy as np
from tqdm import tqdm

tag_to_ind = {}
ind_to_tag = {}
token_to_ind = {}
ind_to_token = {}
vocab_size = 0
output_size = 0

def encodeToken(token):
    vector = np.zeros((vocab_size, 1))
    vector[token_to_ind[token]] = 1
    return vector

def encodeTag(tag):
    vector = np.zeros((output_size, 1))
    vector[tag_to_ind[tag]] = 1
    return vector

def decodeTag(vector):
    ind = np.argmax(vector)
    tag = ind_to_tag[ind]
    return tag

def sigmoid(x, derivative = False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x, derivative = False):
    if derivative:
        return 1 - x**2
    else:
        return np.tanh(np.clip(x, -500, 500))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def initWeight(m, n):
    W = np.random.uniform(-1, 1, (m, n)) * np.sqrt(6 / (m + n))
    return W

def initBias(m , n):
    B = np.zeros((m , n))
    return B

def getData(file_path, size):
    corpus = open(file_path, 'r')
    sentences = []
    sentence = []
    labels = []
    label = []
    cnt = 0

    for line in corpus:
        if line.startswith('-DOCSTART-') or line == "\n":
            if sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
                cnt += 1
                if cnt >= size:
                    break
            continue
        parts = line.strip().split()
        if len(parts) > 1:
            sentence.append(parts[0])
            label.append(parts[-1])

    return sentences, labels

def getDictionaries(tokens, tags):
    global tag_to_ind 
    global ind_to_tag 
    global token_to_ind 
    global ind_to_token
    token_to_ind["PAD"] = 0
    ind_to_token[0] = "PAD"
    for i, token in enumerate(tokens):
        token_to_ind[token] = i + 1
        ind_to_token[i + 1] = token
    
    for i, tag in enumerate(tags):
        tag_to_ind[tag] = i
        ind_to_tag[i] = tag

def padding(sentences, labels, sentence_len):
    for i, sentence in enumerate(sentences):
        if len(sentence) < sentence_len:
            sentence = sentence + ["PAD"] * (sentence_len - len(sentence))
            sentences[i] = sentence
    
    for i, label in enumerate(labels):
        if len(label) < sentence_len:
            label = label + ["O"] * (sentence_len - len(label))
            labels[i] = label
    
    return sentences, labels

def preProcess(sentences, labels):
    global vocab_size
    global output_size
    
    sentence_len = 0
    tokens = set()
    tags = set()
    number_of_sentences = len(sentences)
    
    for sentence in sentences:
        sentence_len = max(sentence_len, len(sentence))
        for words in sentence:
            tokens.add(words)
    vocab_size = len(tokens) + 1

    for label in labels:
        for tag in label:
            tags.add(tag)
    output_size = len(tags)

    getDictionaries(tokens, tags)
    sentences, labels = padding(sentences, labels, sentence_len) 
    
    inputs = np.zeros((number_of_sentences, sentence_len, vocab_size))
    tags = np.zeros((number_of_sentences, sentence_len, output_size))
    for i in range(0, number_of_sentences):
        for j in range(0, sentence_len):
            vector_tag = encodeTag(labels[i][j])
            vector_token = encodeToken(sentences[i][j])
            inputs[i][j] = vector_token.flatten()
            tags[i][j] = vector_tag.flatten()

    return inputs, tags, sentence_len

class LSTM:
    def __init__(self, X, Y, input_size, hidden_size, output_size, sentence_len, vocab_size):
        self.hidden_size = hidden_size
        self.sentence_len = sentence_len
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.Wu = [initWeight(hidden_size, input_size), initWeight(hidden_size, input_size)]
        self.Bu = [initBias(hidden_size, 1), initBias(hidden_size, 1)]
        self.Wf = [initWeight(hidden_size, input_size), initWeight(hidden_size, input_size)]
        self.Bf = [initBias(hidden_size, 1), initBias(hidden_size, 1)]
        self.Wc = [initWeight(hidden_size, input_size), initWeight(hidden_size, input_size)]
        self.Bc = [initBias(hidden_size, 1), initBias(hidden_size, 1)]
        self.Wo = [initWeight(hidden_size, input_size), initWeight(hidden_size, input_size)]
        self.Bo = [initBias(hidden_size, 1), initBias(hidden_size, 1)]
        self.Wy = initWeight(2 * hidden_size, output_size)
        self.By = initBias(output_size, 1)
    
    def save_parameters(self, filename):
        np.savez(
            filename,
            Wu0=self.Wu[0], Wu1=self.Wu[1],
            Bu0=self.Bu[0], Bu1=self.Bu[1],
            Wf0=self.Wf[0], Wf1=self.Wf[1],
            Bf0=self.Bf[0], Bf1=self.Bf[1],
            Wc0=self.Wc[0], Wc1=self.Wc[1],
            Bc0=self.Bc[0], Bc1=self.Bc[1],
            Wo0=self.Wo[0], Wo1=self.Wo[1],
            Bo0=self.Bo[0], Bo1=self.Bo[1],
            Wy=self.Wy, By=self.By
        )
    
    def load_parameters(self, filename):
        data = np.load(filename)
        self.Wu = [data['Wu0'], data['Wu1']]
        self.Bu = [data['Bu0'], data['Bu1']]
        self.Wf = [data['Wf0'], data['Wf1']]
        self.Bf = [data['Bf0'], data['Bf1']]
        self.Wc = [data['Wc0'], data['Wc1']]
        self.Bc = [data['Bc0'], data['Bc1']]
        self.Wo = [data['Wo0'], data['Wo1']]
        self.Bo = [data['Bo0'], data['Bo1']]
        self.Wy = data['Wy']
        self.By = data['By']

    def reset(self):
        default = np.zeros((self.hidden_size, 1))
        self.concat_inputs = [{}, {}]
        self.activation_states = [{-1:default}, {sentence_len:default}]
        self.cell_states = [{-1:default}, {sentence_len:default}]
        self.candidate_gates = [{}, {}]
        self.output_gates = [{}, {}]
        self.forget_gates = [{}, {}]
        self.input_gates = [{}, {}]
    
    def forwardprop(self, sample):
        x = self.X[sample]
        self.reset()
        predictions = []

        for t in reversed(range(self.sentence_len)):
            xt = x[t].reshape((self.vocab_size, 1))
            self.concat_inputs[1][t] = np.concatenate((self.activation_states[1][t + 1], xt))

            gamma_u = self.Wu[1] @ self.concat_inputs[1][t] + self.Bu[1]
            self.input_gates[1][t] = sigmoid(gamma_u)
            gamma_f = self.Wf[1] @ self.concat_inputs[1][t] + self.Bf[1]
            self.forget_gates[1][t] = sigmoid(gamma_f)
            gamma_o = self.Wo[1] @ self.concat_inputs[1][t] + self.Bo[1]
            self.output_gates[1][t] = sigmoid(gamma_o)

            pc_ = self.Wc[1] @ self.concat_inputs[1][t] + self.Bc[1]
            self.candidate_gates[1][t] = tanh(pc_)
            self.cell_states[1][t] = self.input_gates[1][t] * self.candidate_gates[1][t] + self.forget_gates[1][t] * self.cell_states[1][t + 1]
            self.activation_states[1][t] = self.output_gates[1][t] * tanh(self.cell_states[1][t])
        
        for t in range(self.sentence_len):
            xt = x[t].reshape((self.vocab_size, 1))
            self.concat_inputs[0][t] = np.concatenate((self.activation_states[0][t - 1], xt))

            gamma_u = self.Wu[0] @ self.concat_inputs[0][t] + self.Bu[0]
            self.input_gates[0][t] = sigmoid(gamma_u)
            gamma_f = self.Wf[0] @ self.concat_inputs[0][t] + self.Bf[0]
            self.forget_gates[0][t] = sigmoid(gamma_f)
            gamma_o = self.Wo[0] @ self.concat_inputs[0][t] + self.Bo[0]
            self.output_gates[0][t] = sigmoid(gamma_o)

            pc_ = self.Wc[0] @ self.concat_inputs[0][t] + self.Bc[0]
            self.candidate_gates[0][t] = tanh(pc_)
            self.cell_states[0][t] = self.input_gates[0][t] * self.candidate_gates[0][t] + self.forget_gates[0][t] * self.cell_states[0][t - 1]
            self.activation_states[0][t] = self.output_gates[0][t] * tanh(self.cell_states[0][t])

            at = np.concatenate((self.activation_states[0][t], self.activation_states[1][t]))

            z = self.Wy.T @ at + self.By
            prediction = softmax(z)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def backprop(self, errors, learning_rate):
        dWf = [np.zeros_like(self.Wf[0]), np.zeros_like(self.Wf[1])]
        dBf = [np.zeros_like(self.Bf[0]), np.zeros_like(self.Bf[1])]
        dWu = [np.zeros_like(self.Wu[0]), np.zeros_like(self.Wu[1])]
        dBu = [np.zeros_like(self.Bu[0]), np.zeros_like(self.Bu[1])]
        dWo = [np.zeros_like(self.Wo[0]), np.zeros_like(self.Wo[1])]
        dBo = [np.zeros_like(self.Bo[0]), np.zeros_like(self.Bo[1])]
        dWc = [np.zeros_like(self.Wc[0]), np.zeros_like(self.Wc[1])]
        dBc = [np.zeros_like(self.Bc[0]), np.zeros_like(self.Bc[1])]
        da_next = [np.zeros_like(self.activation_states[0][0]), np.zeros_like(self.activation_states[1][0])]
        dc_next = [np.zeros_like(self.cell_states[0][0]), np.zeros_like(self.cell_states[1][0])]
        dWy = np.zeros_like(self.Wy)
        dBy = np.zeros_like(self.By)

        for t in reversed(range(self.sentence_len)):
            error = errors[t]
            error = np.reshape(error, (9, 1))
            at = np.concatenate((self.activation_states[0][t], self.activation_states[1][t]))
            dWy += at @ error.T
            dBy += error
            Wy = self.Wy[:self.hidden_size]
            da = Wy @ error + da_next[0]
            dc = da * self.output_gates[0][t] * tanh(tanh(self.cell_states[0][t]), derivative = True) + dc_next[0]

            dc_ = dc * self.input_gates[0][t]
            dpc_ = dc_ * tanh(self.candidate_gates[0][t], derivative = True)
            dgamma_u = dc * self.candidate_gates[0][t] * sigmoid(self.input_gates[0][t], derivative = True)
            dgamma_f = dc * self.cell_states[0][t - 1] * sigmoid(self.forget_gates[0][t], derivative = True)
            dgamma_o = da * tanh(self.cell_states[0][t]) * sigmoid(self.output_gates[0][t], derivative = True)

            dc_next = dc * self.forget_gates[0][t]
            da_next = self.Wc[0].T @ dpc_ + self.Wu[0].T @ dgamma_u + self.Wf[0].T @ dgamma_f + self.Wo[0].T @ dgamma_o
            da_next = da_next[:self.hidden_size, :]

            dWc[0] += dpc_ @ self.concat_inputs[0][t].T
            dWu[0] += dgamma_u @ self.concat_inputs[0][t].T
            dWf[0] += dgamma_f @ self.concat_inputs[0][t].T
            dWo[0] += dgamma_o @ self.concat_inputs[0][t].T
            dBc[0] += dpc_
            dBu[0] += dgamma_u
            dBf[0] += dgamma_f
            dBo[0] += dgamma_o

        for t in range(self.sentence_len):
            error = errors[t]
            error = np.reshape(error, (9, 1))
            Wy = self.Wy[self.hidden_size:]
            da = Wy @ error + da_next[1]
            dc = da * self.output_gates[1][t] * tanh(tanh(self.cell_states[1][t]), derivative = True) + dc_next[1]

            dc_ = dc * self.input_gates[1][t]
            dpc_ = dc_ * tanh(self.candidate_gates[1][t], derivative = True)
            dgamma_u = dc * self.candidate_gates[1][t] * sigmoid(self.input_gates[1][t], derivative = True)
            dgamma_f = dc * self.cell_states[1][t + 1] * sigmoid(self.forget_gates[1][t], derivative = True)
            dgamma_o = da * tanh(self.cell_states[1][t]) * sigmoid(self.output_gates[1][t], derivative = True)

            dc_next = dc * self.forget_gates[1][t]
            da_next = self.Wc[1].T @ dpc_ + self.Wu[1].T @ dgamma_u + self.Wf[1].T @ dgamma_f + self.Wo[1].T @ dgamma_o
            da_next = da_next[:self.hidden_size, :]

            dWc[1] += dpc_ @ self.concat_inputs[1][t].T
            dWu[1] += dgamma_u @ self.concat_inputs[1][t].T
            dWf[1] += dgamma_f @ self.concat_inputs[1][t].T
            dWo[1] += dgamma_o @ self.concat_inputs[1][t].T
            dBc[1] += dpc_
            dBu[1] += dgamma_u
            dBf[1] += dgamma_f
            dBo[1] += dgamma_o

        dWy = np.clip(dWy, -1, 1)
        dBy = np.clip(dBy, -1, 1)
        for i in range(0, 2):
            dWo[i] = np.clip(dWo[i], -5, 5)
            dBo[i] = np.clip(dBo[i], -5, 5)
            dWf[i] = np.clip(dWf[i], -5, 5)
            dBf[i] = np.clip(dBf[i], -5, 5)
            dWu[i] = np.clip(dWu[i], -5, 5)
            dBu[i] = np.clip(dBu[i], -5, 5)
            dWc[i] = np.clip(dWc[i], -5, 5)
            dBc[i] = np.clip(dBc[i], -5, 5)
        
        self.Wy += dWy * learning_rate
        self.By += dBy * learning_rate
        for i in range(0, 2):
            self.Wf[i] += dWf[i] * learning_rate
            self.Bf[i] += dBf[i] * learning_rate
            self.Wu[i] += dWu[i] * learning_rate
            self.Bu[i] += dBu[i] * learning_rate
            self.Wo[i] += dWo[i] * learning_rate
            self.Bo[i] += dBo[i]* learning_rate
            self.Wc[i] += dWc[i] * learning_rate
            self.Bc[i] += dBc[i]* learning_rate
    
    def test(self, X_test, Y_test):
        accuracy = 0
        self.load_parameters('lstm_weights_biases.npz')
        for i in range(len(X_test)):
                predictions = self.forwardprop(i)
                predictions = np.reshape(predictions, (self.sentence_len, self.output_size))
                correct = 0
                for j in range(self.sentence_len):
                    prediction = np.argmax(predictions[i])
                    truth = np.argmax(Y_test[i][j])
                    print(Y_test[i][j])
                    token = np.argmax(X_test[i][j])
                    print("-------------------------------------")
                    print("Word : ", ind_to_token[token])
                    print("Model Predicted : ", ind_to_tag[prediction])
                    print("Ground Truth : ", ind_to_tag[truth])
                    print("-------------------------------------")
                    if(ind_to_tag[prediction] == ind_to_tag[truth]):
                        correct += 1
                accuracy += correct / self.sentence_len
        avg_accuracy = accuracy / len(X_test)
        print(avg_accuracy) 
                
    def train(self, epochs, learning_rate):
        for _ in tqdm(range(epochs)):
            total_loss = 0
            last_loss = 100000
            for i in range(len(self.X)):
                predictions = self.forwardprop(i)
                predictions = np.reshape(predictions, (self.sentence_len, self.output_size))
                errors = self.Y[i] - predictions
                predictions = np.clip(predictions, 1e-12, 1. - 1e-12)
                loss = -np.sum(self.Y[i] * np.log(predictions)) / self.Y[i].shape[0]

                total_loss += loss
                self.backprop(errors, learning_rate)
            avg_loss = total_loss
            if last_loss > avg_loss:
                last_loss = avg_loss
                self.save_parameters('lstm_weights_biases.npz')

            print(avg_loss)
    
if __name__ == "__main__":
    size = 50
    split = int(size * 0.8)
    sentences, labels = getData("train.txt", size)
    inputs, tags, sentence_len = preProcess(sentences, labels)
    X_train = inputs[:split]
    Y_train = tags[:split]
    print(X_train.shape)
    X_test = inputs[split:]
    Y_test = tags[split:]
    hidden_size = 150
    lstm = LSTM(X_train, Y_train, vocab_size + hidden_size, hidden_size, output_size, sentence_len, vocab_size)
    lstm.train(50, 0.1)
    lstm.test(X_test, Y_test)