
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch

PATH_DATA = 'input.txt'

def encode(unique_chars: list, input: str):
    stoi = {ch:i for i, ch in enumerate(unique_chars)}
    return [stoi[c] for c in input] # encoder: take a string, output a list of integers

def decode(unique_chars: list, input: list):
    itos = {i:ch for i, ch in enumerate(unique_chars)}
    return ''.join([itos[i] for i in input]) # decoder: take a list of integers, output a string

if __name__ == '__main__':

    f = open(PATH_DATA, 'r', encoding='utf-8')
    text = f.read()
    f.close()

    print('Length dataset in characters:', len(text), '\n')

    print('Sample input text:\n', text[:200], '\n')

    unique_chars = sorted(list(set(text)))
    print('Unique characters:', ''.join(unique_chars))
    print('Length unique characters:', len(unique_chars), '\n')

    # encode data
    data = torch.tensor(encode(unique_chars, text), dtype=torch.long)
    print('Encoded data shape:', data.shape, 'Encoded data type:', data.dtype, '\n')

    print('Sample input encoded text:\n', data[:200], '\n')
