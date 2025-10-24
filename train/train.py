
import torch

import train.train_config as train_config

def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - train_config.context_length, (train_config.batch_size,))
    x = torch.stack([data[i:i+train_config.context_length] for i in ix]) # stack input chunks (rows)
    y = torch.stack([data[i+1:i+train_config.context_length+1] for i in ix]) # stack targets (rows)
    # shape of x and y will be (config.batch_size, config.context_length)
    return x, y

if __name__ == '__main__':
    pass
