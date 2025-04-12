import json
import time
import numpy as np
import regex as re 
import multiprocessing
import pickle

from tokenizer import BPETokenizer

if __name__ == '__main__':

    bpe2 = BPETokenizer([], [], ["<|endoftext|>"])
    encoded = bpe2.encode_from_pretokens(
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_vocab_0411.pickle',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_merges_0411.pickle', 
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_index_to_list_0411.pickle',
        '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_pretoken_index_0411.pickle',
        ["<|endoftext|>"]
    )
    # print(encoded)

    print(bpe2.decode(encoded[-200:]))
    print(encoded[-200:])

    # do standard encoding 

    # bpe1 = BPETokenizer([], [], ["<|endoftext|>"])
    # bpe1.from_files(
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_vocab_0411.pickle',
    #     '/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_merges_0411.pickle', 
    #     ["<|endoftext|>"]
    # )

    # # load tiny stories vocab
    # with open('/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt', 'r') as file:
    #     tinystories_valid = file.read()

    # print(bpe1.encode(tinystories_valid))

    