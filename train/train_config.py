
# when training we do not input the full dataset into the transformer, we sample random chunks (context) of a certain **maximum** length
# actually, when selecting a random chunk we do train_data[:context_length+1]
# context_length+1 because:
    # a chunk of length 8 has multiple examples into it: we will train the transformer simultaneously for every position of the chunk
    # example: chunk (context) = [18, 47, 56, 57, 58, 1, 15, 47, 58]
    # in a chunk of 9 elements, there are 8 (context_length) different examples:
        # context -> prediction
        # 18 -> 47
        # 18, 47 -> 56
        # 18, 47, 56 -> 57
        # 18, 47, 56, 57 -> 58
        # ...
    # in this way, in inference time the model will be able to do predictions given an input of length between 1 and context_length
    # if the input context is longer than context_length -> we need to trunkate
context_length = 8 # chunk length

# we are going to have batches of chunks: GPUs are very good at parallel processing of data
# the chunks in a batch are processed by the model completely independently
batch_size = 4


