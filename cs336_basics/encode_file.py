import json
import time
import numpy as np
import regex as re 
import multiprocessing
import pickle

from tokenizer import BPETokenizer

if __name__ == '__main__':

    bpe2 = BPETokenizer(vocab={}, merges={}, special_tokens=["<|endoftext|>"])
    # encoded = bpe2.encode_from_pretokens(
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt',
    #     # '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/owt_10docs.txt',
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_vocab_0411.pickle',
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_merges_0411.pickle', 
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_index_to_list_0411.pickle',
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_pretoken_index_0411.pickle',
    #     ["<|endoftext|>"]
    # )
    encoded = bpe2.encode_from_pretokens(
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/owt_train.txt',
        # '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/owt_10docs.txt',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/owt_train_v32000_vocab_0411.pickle',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/owt_train_v32000_merges_0411.pickle', 
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/owt_train_v32000_index_to_list_0411.pickle',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/owt_train_v32000_pretoken_index_0411.pickle',
        ["<|endoftext|>"]
    )
    # print(encoded)

    my_array = np.array(encoded, dtype=np.uint16)

    # Save the NumPy array to a .npy file
    np.save('OWTTrain_tokenized.npy', my_array)

    # print(bpe2.decode(encoded[-200:]))
    # print(encoded[-200:])
    