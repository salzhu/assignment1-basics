import json
import time
import numpy as np
import regex as re 
import multiprocessing

from tokenizer import train_bpe

def worker(args):
    return train_bpe(args[0], args[1], args[2])

if __name__ == '__main__':

    input_path = '/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    #input_path = '/Users/sallyzhu/Desktop/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt'
    # pool = multiprocessing.Pool()

    # ocab, merges = pool.map(worker, [(input_path, 10000, "<|endoftext|>")])

    train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )