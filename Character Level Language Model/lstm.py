import numpy as np
from tqdm import tqdm

def oneHotEncode(text):
    vector = np.zeros((char_size, 1))
    vector[char_to_ind[text]] = 1

    return vector

def initWeights(m , n):
    W = np.random.uniform(-1, 1, (m, n)) * np.sqrt(6 / (m + n))
    return W

def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)
    else:
        return 1 / (1 + np.exp(-input))

def tanh(input, derivative = False):
    if derivative:
        return 1 - input**2
    else:
        return np.tanh(input)

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        self.Wu = initWeights(hidden_size, input_size)
        self.Bu = np.zeros((hidden_size, 1))
        self.Wf = initWeights(hidden_size, input_size)
        self.Bf = np.zeros((hidden_size, 1))
        self.Wc = initWeights(hidden_size, input_size)
        self.Bc = np.zeros((hidden_size, 1))
        self.Wo = initWeights(hidden_size, input_size)
        self.Bo = np.zeros((hidden_size, 1))
        self.Wy = initWeights(hidden_size, output_size)
        self.By = np.zeros((output_size, 1))

    def reset(self):
        self.concat_inputs = {}
        self.activation_states = {-1:np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1:np.zeros((self.hidden_size, 1))}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}

    def forward(self, inputs):
        self.reset()
        predictions = []

        for t in range(len(inputs)):
            self.concat_inputs[t] = np.concatenate((self.activation_states[t - 1], inputs[t]))

            gamma_u = self.Wu @ self.concat_inputs[t] + self.Bu
            self.input_gates[t] = sigmoid(gamma_u)
            gamma_f = self.Wf @ self.concat_inputs[t] + self.Bf
            self.forget_gates[t] = sigmoid(gamma_f)
            gamma_o = self.Wo @ self.concat_inputs[t] + self.Bo
            self.output_gates[t] = sigmoid(gamma_o)

            pc_ = self.Wc @ self.concat_inputs[t] + self.Bc
            self.candidate_gates[t] = tanh(pc_) 
            self.cell_states[t] = self.input_gates[t] * self.candidate_gates[t] + self.forget_gates[t] * self.cell_states[t - 1]
            self.activation_states[t] = self.output_gates[t] * tanh(self.cell_states[t])
            z = self.Wy.T @ self.activation_states[t] + self.By
            prediction = softmax(z)
            predictions.append(prediction)

        return predictions

    def backward(self, errors, inputs, learning_rate):
        dWf = np.zeros_like(self.Wf)
        dBf = np.zeros_like(self.Bf)
        dWu = np.zeros_like(self.Wu)
        dBu = np.zeros_like(self.Bu)
        dWo = np.zeros_like(self.Wo)
        dBo = np.zeros_like(self.Bo)
        dWc = np.zeros_like(self.Wc)
        dBc = np.zeros_like(self.Bc)
        dWy = np.zeros_like(self.Wy)
        dBy = np.zeros_like(self.By)
        da_next = np.zeros_like(self.activation_states[0])
        dc_next = np.zeros_like(self.cell_states[0])

        for t in reversed(range(len(inputs))):
            error = errors[t]

            dWy += self.activation_states[t] @ error.T
            dBy += error

            da = self.Wy @ error + da_next
            dc = da * self.output_gates[t] * tanh(tanh(self.cell_states[t]), derivative = True) + dc_next

            dc_ = dc * self.input_gates[t]
            dpc_ = dc_ * tanh(self.candidate_gates[t], derivative = True)
            dgamma_u = dc * self.candidate_gates[t] * sigmoid(self.input_gates[t], derivative = True)
            dgamma_f = dc * self.cell_states[t - 1] * sigmoid(self.forget_gates[t], derivative = True)
            dgamma_o = da * tanh(self.cell_states[t]) * sigmoid(self.output_gates[t], derivative = True)

            dc_next = dc * self.forget_gates[t]
            da_next = self.Wc.T @ dpc_ + self.Wu.T @ dgamma_u + self.Wf.T @ dgamma_f + self.Wo.T @ dgamma_o
            da_next = da_next[:self.hidden_size, :]

            dWc += dpc_ @ self.concat_inputs[t].T
            dWu += dgamma_u @ self.concat_inputs[t].T
            dWf += dgamma_f @ self.concat_inputs[t].T
            dWo += dgamma_o @ self.concat_inputs[t].T
            dBc += dpc_
            dBu += dgamma_u
            dBf += dgamma_f
            dBo += dgamma_o

        dWy = np.clip(dWy, -1, 1)
        dBy = np.clip(dBy, -1, 1)
        dWo = np.clip(dWo, -1, 1)
        dBo = np.clip(dBo, -1, 1)
        dWf = np.clip(dWf, -1, 1)
        dBf = np.clip(dBf, -1, 1)
        dWu = np.clip(dWu, -1, 1)
        dBu = np.clip(dBu, -1, 1)
        dWc = np.clip(dWc, -1, 1)
        dBc = np.clip(dBc, -1, 1)

        self.Wf += dWf * learning_rate
        self.Bf += dBf * learning_rate
        self.Wu += dWu * learning_rate
        self.Bu += dBu * learning_rate
        self.Wo += dWo * learning_rate
        self.Bo += dBo * learning_rate
        self.Wc += dWc * learning_rate
        self.Bc += dBc * learning_rate
        self.Wy += dWy * learning_rate
        self.By += dBy * learning_rate

    def train(self, inputs, labels, epochs, learning_rate):
        inputs = [oneHotEncode(input) for input in inputs]

        for _ in tqdm(range(epochs)):
            predictions = self.forward(inputs)

            errors = []
            for q in range(len(predictions)):
                errors += [-predictions[q]]
                errors[-1][char_to_ind[labels[q]]] += 1

            self.backward(errors, self.concat_inputs, learning_rate)    
        
    def test(self, inputs, labels):
        accuracy = 0
        probabilities = self.forward([oneHotEncode(input) for input in inputs])

        output = ''
        for q in range(len(labels)):
            prediction = ind_to_char[np.random.choice([*range(char_size)], p = probabilities[q].reshape(-1))]

            output += prediction

            if prediction == labels[q]:
                accuracy += 1

        print(f'Ground Truth:\nt{labels}\n')
        print(f'Predictions:\nt{"".join(output)}\n')
        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)}%')   



if __name__ == "__main__":
    data = """To be, or not to be, that is the question: Whether \
    'tis nobler in the mind to suffer The slings and arrows of ou\
    trageous fortune, Or to take arms against a sea of troubles A\
    nd by opposing end them. To die—to sleep, No more; and by a s\
    leep to say we end The heart-ache and the thousand natural sh\
    ocks That flesh is heir to: 'tis a consummation Devoutly to b\
    e wish'd. To die, to sleep; To sleep, perchance to dream—ay, \
    there's the rub: For in that sleep of death what dreams may c\
    ome, When we have shuffled off this mortal coil, Must give us\
    pause—there's the respect That makes calamity of so long lif\
    e. For who would bear the whips and scorns of time, Th'oppres\
    sor's wrong, the proud man's contumely, The pangs of dispriz'\
    d love, the law's delay, The insolence of office, and the spu\
    rns That patient merit of th'unworthy takes, When he himself \
    might his quietus make""".lower()

    vocabulary = set(data)
    data_size = len(data)
    char_size = len(vocabulary)

    char_to_ind = {c:i for i, c in enumerate(vocabulary)}
    ind_to_char = {i:c for i, c in enumerate(vocabulary)}

    train_X = data[:-1]
    train_Y = data[1:]

    hidden_size = 25

    lstm = LSTM(char_size + hidden_size, hidden_size, char_size)
    lstm.train(train_X, train_Y, 500, 0.05)
    lstm.test(train_X, train_Y)
    


