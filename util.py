from collections import Counter
from keras import backend as K

def vocab(path, vocab_size, EOS=False):
    with open(path, mode="r") as f1:
        token = []
        for line in f1:
            if EOS:
                word = line.strip().split(" ") + ["</s>"] + ["<EOS>"]
            else:
                word = line.strip().split(" ") + ["</s>"]
            for w in word:
                token.append(w)
        vocab = Counter(token).most_common(vocab_size)
        vocab, _ = zip(*vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(vocab))
        idx_word = dict((value, key) for key, value in word_idx.items())
        return word_idx, idx_word


def word2idx(word_idx, sentence):
    unk = len(word_idx) + 1
    return [unk if word not in word_idx else word_idx[word] for word in sentence]


def sentence2idx(path, vocab_size=False, word_idx=False, train=False):
    # convert sentence words into idx using given word_idx dictionary
    if not vocab_size and word_idx:
        with open(path, mode="r") as f:
            if train:
                sentences = [word2idx(word_idx, ["</s>"]+line.strip().split()+["<EOS>"]) for line in f]
            else:
                sentences = [word2idx(word_idx, line.strip().split()+["</s>"]) for line in f]
        return sentences, False, False

    elif vocab_size:
        word_idx, idx_word = vocab(path, vocab_size, EOS=train)
        with open(path, mode="r") as f:
            if train:
                sentences = [["</s>"]+line.strip().split()+["<EOS>"] for line in f]
            else:
                sentences = [line.strip().split()+["</s>"] for line in f]
        sentences = [word2idx(word_idx, sentence) for sentence in sentences]
        word_idx['<unk>'] = len(word_idx)
        idx_word[len(word_idx)] = '<unk>'
        return sentences, word_idx, idx_word

    else:
        raise TypeError
        
'''
functions for Keras Lambda Layer
'''
# get last layer of encoder
def last_layer(x):
    return x[:,-1,:]
# return the output shape
def last_layer_output(input_shape):
    return (input_shape[0], input_shape[2])

# get sum of vector/tensor
def sum_tensor(x):
    return K.sum(x, axis=1)

# return the output shape
def sum_tensor_output(input_shape):
    return (input_shape[0], input_shape[2])
