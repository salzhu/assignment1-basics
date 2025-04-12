import json
import time
import numpy as np
import regex as re 
import multiprocessing
import pickle

from tokenizer import train_bpe_clean_splits_set as train_bpe

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/')
parser.add_argument('--file', type=str, default='TinyStoriesV2-GPT4-train')
parser.add_argument('--date', type=str, default='0411')
parser.add_argument('--vocab_size', type=int, default=10000)
args = parser.parse_args()

if __name__ == '__main__':

    vocab, merges, index_to_list_count, pretokens_to_index = train_bpe(
        input_path=f'{args.path}{args.file}.txt',
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    with open(f'{args.file}_v{args.vocab_size}_vocab_{args.date}.pickle', 'wb') as file:
        pickle.dump(vocab, file)
    with open(f'{args.file}_v{args.vocab_size}_merges_{args.date}.pickle', 'wb') as file:
        pickle.dump(merges, file)
    with open(f'{args.file}_v{args.vocab_size}_index_to_list_{args.date}.pickle', 'wb') as file:
        pickle.dump(index_to_list_count, file)
    with open(f'{args.file}_v{args.vocab_size}_pretoken_index_{args.date}.pickle', 'wb') as file:
        pickle.dump(pretokens_to_index, file)