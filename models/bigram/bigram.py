
import torch
import torch.nn.functional as F

class Bigram:

    vocab: list = None # vocabulary
    stoi:dict = None # string to int
    itos: dict = None # int to string
    N: torch.tensor = None # table with counts
    P: torch.tensor = None # table with probabilities
    W: torch.tensor = None # table with weights (NN)

    def __init__(self, ):
        pass

    def extract_info_from_train_data(self, train_data):
        self.vocab = sorted(list(set(train_data)))
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

    # COUNTS APPROACH

    def train_counts(self, train_data: str, smoothing: int):

        self.extract_info_from_train_data(train_data)

        # N is a table of len(self.vocab) * len(self.vocab), where rows represent a character and columns the following character
        # in each position of the table there is stored how many times the character 'row' is followed by the character 'col' in the train set
        self.N = torch.zeros((len(self.vocab), len(self.vocab)), dtype=torch.int32)
        
        # just count
        for ch1, ch2 in zip(train_data, train_data[1:]):
            i1 = self.stoi[ch1]
            i2 = self.stoi[ch2]
            self.N[i1, i2] += 1
        
        # by normalizing each row we get probabilities, then we can sample for each character a following one using the corresponding distribution
        # smoothing is an integer that will be added to all the counts in N, in this way we are smoothing the distribution
        # it is useful when we have pairs of characters with 0 ocurrences, which can lead us to inf results (when applying log, see loss function)
        self.P = (self.N+smoothing).float()
        # inplace operation because of /= (we avoid creating a new tensor)
        # caution with dims when broadcasting! keepdims really important to be True
        self.P /= self.P.sum(1, keepdims=True)

    def infer_counts(self, context: str, generator: torch.Generator = None):

        out = []
        i = self.stoi[context[-1]] # because it is a bigram model, we only care about the last character in the context
        while True:
            p = self.P[i]
            # sample next character from corresponding distribution
            # generator for reproducibility
            i = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(self.itos[i])
            if i == self.stoi['\n']: # TODO
                break
        
        return ''.join(out)
    
    def calc_loss_given_data_counts(self, data: str):

        # we want a number that tells us how good the prediction is
        
        # likelihood: product of probabilities
        # but, there are a lot of probabilities close to zero! -> likelihood very small
        # log_likelihood: we can do log(likelihood) to get more manageble numbers (bigger numbers in absolute value)
        # however, log(likelihood) is negative (or 0 at most)
        # negative_log_likelihood: we can do -log(likelihood) to get positive numbers
        # the smaller the likelihood, the larger the -log(likelihood) -> it behaves like a loss!
        # recall: let a,b,c be probs: likelihood = a*b*c, log(a*b*c) = log(a) + log(b) + log(c)
        # we can do average instead of the sum

        # SUMMARY:
        # GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
        # (in other words, change the model parameters in order to maximize likelihood of the prediction)
        # equivalent to maximizing the log likelihood (because log is monotonic)
        # equivalent to minimizing the negative log likelihood
        # equivalent to minimizing the average negative log likelihood

        log_likelihood = 0.0
        n = 0

        for ch1, ch2 in zip(data, data[1:]):
            i1 = self.stoi[ch1]
            i2 = self.stoi[ch2]
            prob = self.P[i1, i2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
        
        return -log_likelihood/n

    # NN APPROACH (1 linear layer of len(self.vocab) neurons)

    def train_nn(self, train_data: str, epochs: int, learning_rate: float, regularization: float,  generator: torch.Generator = None):

        self.extract_info_from_train_data(train_data)

        xs, ys = [], [] # x is first character (input) and y second character (label)

        for ch1, ch2 in zip(train_data, train_data[1:]):
            i1 = self.stoi[ch1]
            i2 = self.stoi[ch2]
            xs.append(i1)
            ys.append(i2)

        xs = torch.tensor(xs) # batch of inputs
        ys = torch.tensor(ys) # batch of labels

        # random initialization of weights
        # see explanation of self.W shape in self.forward_pass_nn function
        self.W = torch.randn((len(self.vocab), len(self.vocab)), generator=generator, requires_grad=True) # returns a tensor filled with random numbers from a standard normal distribution
        # 'requires_grad' important for backpropagation!

        for e in range(epochs):

            print('---- Epoch {} ----'.format(e))

            # forward pass
            _, loss = self.forward_pass_nn(xs, ys, regularization)
            
            # backward pass
            self.W.grad = None # set to zero the gradient
            loss.backward()
            
            # update
            self.W.data += -learning_rate * self.W.grad # gradient descent

            print('Loss:', loss.item())
    
    def infer_nn(self, context: str, generator: torch.Generator = None):

        out = []
        i = self.stoi[context[-1]] # because it is a bigram model, we only care about the last character in the context
        while True:
            p, _ = self.forward_pass_nn(xs=torch.tensor([i]))
            # sample next character from corresponding distribution
            # generator for reproducibility
            i = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(self.itos[i])
            if i == self.stoi['\n']: # TODO
                break
        
        return ''.join(out)

    def calc_loss_given_data_nn(self, data: str, regularization: float):

        xs, ys = [], [] # x is first character (input) and y second character (label)

        for ch1, ch2 in zip(data, data[1:]):
            i1 = self.stoi[ch1]
            i2 = self.stoi[ch2]
            xs.append(i1)
            ys.append(i2)

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        return self.forward_pass_nn(xs, ys, regularization)[1]

    def forward_pass_nn(self, xs: torch.tensor, ys: torch.tensor = None, regularization: float = None):

        # for each batch of inputs shape=(X, len(self.vocab)) where X is the batch_size and each input (row) is a char one hot encoded
        # we want an output shape=(X, len(self.vocab)) -> for each input a probability distribution (a prob for each possible next char)
        # therefore, we need the matrix self.W (weights of our NN) to have shape (len(self.vocab), len(self.vocab))
        # to get the output, we only need to do a matrix multiplication between the batch of inputs (X, len(self.vocab)) and self.W (len(self.vocab), len(self.vocab))
        # the output distribution for each input is the activation value of the len(self.vocab) neurons (it is like selecting the corresponding row of counts in the counts approach)
        # our NN is 1 linear layer of len(self.vocab) neurons

        # the NN output for an input has to be a probability distribution
        # we need some way of transforming the output of the NN (now, it can contain negative values, larger than 1 values, etc.) to a probability distribution
        # how we interpret the output of the NN for a given input:
            # we will see the output as 'log counts' (also called logits)
            # to get the 'counts', we need to exponentiate them (e^x)
            # the probabilities are the counts normalized
            # basically, softmax
        # all of these are differential operantions that we can backpropagate!

        xenc = F.one_hot(xs, num_classes=len(self.vocab)).float() # input to the network: one-hot encoding
        logits = xenc @ self.W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

        if not ys == None and not regularization == None:
            # vectorized loss (average negative log likelihood) + regularization (equivalent as smoothing)
            loss = -probs[torch.arange(len(xs)), ys].log().mean() + regularization*(self.W**2).mean()
            # regularization note: if self.W all 0s -> we get a uniform distribution in output, that's why we can see regularization as smoothing
            return probs, loss
        else:
            return probs, None


def main_counts():

    model = Bigram()
    train_data = open('data/input.txt', 'r').read()
    g = torch.Generator().manual_seed(123)
    model.train_counts(train_data, 1)

    input = 'hello'

    print('Loss value of all training set:', model.calc_loss_given_data_counts(train_data).item())
    print('Input:', input)

    print('----')
    for _ in range(10):
        output_data = model.infer_counts(input, g)
        print('Output:', output_data)
        print('Loss value for above output:', model.calc_loss_given_data_counts(output_data).item())
        print('----')

def main_weights():

    model = Bigram()
    train_data = open('data/input.txt', 'r').read()
    g = torch.Generator().manual_seed(123)
    model.train_nn(train_data=train_data, epochs=10, learning_rate=50, regularization=0.01, generator=g)
    
    input = 'hello'

    print('Loss value of all training set:', model.calc_loss_given_data_nn(data=train_data, regularization=0.01).item())
    print('Input:', input)

    print('----')
    for _ in range(10):
        output_data = model.infer_nn(input, g)
        print('Output:', output_data)
        print('Loss value for above output:', model.calc_loss_given_data_nn(data=output_data, regularization=0.01).item())
        print('----')

if __name__ == '__main__':

    main_counts()
    # main_weights()
