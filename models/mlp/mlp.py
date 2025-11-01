
import torch
import torch.nn.functional as F

from utils.utils import extract_info_from_train_data

class MLP:

    vocab: list = None # vocabulary
    stoi:dict = None # string to int
    itos: dict = None # int to string
    C: torch.tensor = None # look-up table of embeddings
    W1: torch.tensor = None # hidden layer weights
    b1: torch.tensor = None # hidden layer bias
    W2: torch.tensor = None # output layer weights
    b2: torch.tensor = None # output layer bias
    params: list = None # list of torch.tensors containing all the parameters
    num_params: int = None # total number of params
    context_length: int = None # context length
    embedding_dim: int = None # embedding number of dimensions

    def __init__(self, ):
        pass

    def build_dataset(self, raw_data):
        X, Y = [], []
        context = [self.stoi[ch] for ch in raw_data[0:self.context_length]]
        for ch in raw_data[self.context_length:-self.context_length]:
            ix = self.stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

        X = torch.tensor(X) # batch of inputs (chunks from train_data of size self.context_length)
        Y = torch.tensor(Y) # batch of labels

        return X, Y

    def train(self, train_data: str, epochs: int, learning_rate: float, batch_size: int, context_length: int, embedding_dim: int,
              number_neurons_hidden_layer: int,  generator: torch.Generator = None, random_init: bool = True, val_percentage: float = 0.1):
        
        # initialize everything
        if random_init:
            self.vocab, self.stoi, self.itos = extract_info_from_train_data(train_data)
            self.context_length = context_length
            self.embedding_dim = embedding_dim
            self.C =  torch.randn((len(self.vocab), self.embedding_dim), generator=generator)
            self.W1 = torch.randn((self.context_length*self.embedding_dim, number_neurons_hidden_layer), generator=generator)
            self.b1 = torch.randn((number_neurons_hidden_layer), generator=generator)
            self.W2 = torch.randn((number_neurons_hidden_layer, len(self.vocab)), generator=generator)
            self.b2 = torch.randn((len(self.vocab)), generator=generator)
            self.params = [self.C, self.W1, self.b1, self.W2, self.b2]
            self.num_params = sum(p.nelement() for p in self.params)

        # data
        split = int((1-val_percentage)*len(train_data))
        X_train, Y_train = self.build_dataset(train_data[:split])
        X_val, Y_val = self.build_dataset(train_data[split:])

        # training
        for p in self.params:
            p.requires_grad = True # important

        for e in range(epochs):

            print('---- Epoch {} ----'.format(e))

            # minibatch construct
            ix = torch.randint(0, X_train.shape[0], (batch_size,))
        
            # FORWARD PASS
            _, loss_train = self.forward_pass(X_train[ix], Y_train[ix])

            # BACKWARD PASS
            # ensure that all gradients are 0
            for p in self.params:
                p.grad = None
            # populate the gradients
            loss_train.backward()
            # update
            for p in self.params:
                p.data += -learning_rate * p.grad
            
            print('Loss Train (minibatch):', loss_train.item())
            print('Loss Val:', self.forward_pass(X_val, Y_val)[1].item())

    def forward_pass(self, X: torch.tensor, Y: torch.tensor = None):

        # X shape (N, self.context_length) being N the number of examples in the tran_data with len = self.context_length
        # emb shape (N, self.context_length, self.embedding_dim)
        emb = self.C[X] # we can index with torch tensors directly!!

        # we want to do emb @ self.W1 + self.b1, but first we new to reshape the inputs (concatenate the embeddings)
        # emb shape (N, self.context_length*self.embedding_dim)
        h = torch.tanh(emb.view(-1, self.context_length*self.embedding_dim) @ self.W1 + self.b1) # tanh!

        # see explanation of this in bigram model!
        logits = h @ self.W2 + self.b2
        # counts = logits.exp()
        # probs = counts / counts.sum(1, keepdims=True)
        # loss = -probs[torch.arange(X.shape[0]), Y].log().mean()
        # it can be done with a built-in function that does the same!
        probs = F.softmax(logits, dim=1)
        if not Y is None:
            loss = F.cross_entropy(logits, Y) # more efficient and numerically well-behaved
            return probs, loss
        return probs, None

    def infer(self, context: str, generator: torch.Generator = None):
    
        out = []
        context_aux = [self.stoi[ch] for ch in context[0:self.context_length]] # what the model can actually take as context

        while True:
            p, _ = self.forward_pass(X=torch.tensor([context_aux]))
            # sample next character from corresponding distribution
            # generator for reproducibility
            i = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(self.itos[i])
            context_aux = context_aux[1:] + [i] # update context_aux
            if i == self.stoi['\n']: # TODO
                break
        
        return ''.join(out)
    
def main():
    model = MLP()
    train_data = open('data/input.txt', 'r').read()
    generator = torch.Generator().manual_seed(123)
    model.train(train_data=train_data, epochs=100,
                learning_rate=0.1, batch_size=32, context_length=5,
                embedding_dim=3, number_neurons_hidden_layer=300, generator=generator)
    
    print(model.infer('hello', generator))

if __name__ == '__main__':
    main()

# TODO: plot embeddings!
# TODO: grid/random search of hyperparameters!
