
def extract_info_from_train_data(train_data):
    vocab = sorted(list(set(train_data)))
    stoi = {s:i for i,s in enumerate(vocab)}
    itos = {i:s for s,i in stoi.items()}
    return vocab, stoi, itos