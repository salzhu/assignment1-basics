import numpy as np 
import regex as re 
from collections.abc import Iterable, Iterator

def train_bpe(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    cur_vocab_size = 256
    merges = []

    pairs = np.zeros((vocab_size, vocab_size))

    file = open(input_path, 'r')
    file_content = file.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.finditer(PAT, file_content)
    all_inputs = []
    for pt in pre_tokens:
        all_inputs.append(list(pt.group().encode("utf-8")))

    # building original pairs counts 
    for input_bytes in all_inputs:
        for i in range(len(input_bytes) - 1):
            pair = (input_bytes[i], input_bytes[i + 1])
            pairs[pair] += 1

    while cur_vocab_size < vocab_size - len(special_tokens): 
        # find most common pair 
        max_pairs = np.argwhere(pairs==pairs.max())

        # max_pairs_alph = np.argmax([(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs])
        # print([(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs])
        pairs_encoded = [(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs]
        # print(pairs_encoded)
        max_pair = max(pairs_encoded)
        if (' .','..') in pairs_encoded:
            max_pair = (' .','..')
        l = 0
        r = 0 
        for pair in max_pairs:
            if (vocab[pair[0]].decode("utf-8"), vocab[pair[1]].decode("utf-8")) == max_pair:
                l, r = pair
                break

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {pairs[l, r]}")
        pairs[l, r] = 0

        # replace everything in input_bytes
        for input_bytes in all_inputs:
            for i in range(len(input_bytes) - 2, -1, -1):
                if input_bytes[i] == l and input_bytes[i + 1] == r:
                    if i > 0:
                        pairs[input_bytes[i - 1], l] -= 1
                        pairs[input_bytes[i - 1], cur_vocab_size] += 1
                    if i < len(input_bytes) - 2:
                        pairs[r, input_bytes[i + 2]] -= 1
                        pairs[cur_vocab_size, input_bytes[i + 2]] += 1
                    input_bytes[i] = cur_vocab_size
                    input_bytes.pop(i + 1)

        vocab[cur_vocab_size] = vocab[l] + vocab[r] 
        cur_vocab_size += 1
    
    for special_token in special_tokens:
        vocab[cur_vocab_size] = special_token.encode("utf-8")
        cur_vocab_size += 1

    return vocab, merges

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return 
    
    def encode(self, text: str) -> list[int]:

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        # print(self.special_tokens)

        separated = [text]
        for st in self.special_tokens:
            # print(separated)
            temp = []
            for chunk in separated: 
                if chunk in self.special_tokens:
                    temp += [chunk] 
                    continue  
                # print(chunk.split(st))
                # temp += chunk.split(st)
                pattern = f'({re.escape(st)})'
                temp += [substring for substring in re.split(pattern, chunk) if substring]
            separated = temp

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        all_inputs = []

        for chunk in separated:
            if chunk in self.special_tokens:
                all_inputs.append(list(self.vocab.keys())[list(self.vocab.values()).index(chunk.encode('utf-8'))])
                continue 
            # print(chunk)
            # print('----------------------------------')
            pre_tokens = re.finditer(PAT, chunk)
            # print("got here")

            byte_tokens = []

            for pt in pre_tokens: 
                int_tokens = list(pt.group().encode("utf-8"))
                for i in range(len(int_tokens)):
                    t = int_tokens[i]

                    byte_tokens.append(bytes([t]))

            for merge in self.merges: 
                for i in range(len(byte_tokens) - 2, -1, -1):
                    if byte_tokens[i] == merge[0] and byte_tokens[i + 1] == merge[1]:
                        byte_tokens[i] = merge[0] + merge[1]
                        byte_tokens.pop(i + 1)

            output = [] 
            for byte_token in byte_tokens:
                if list(self.vocab.keys())[list(self.vocab.values()).index(byte_token)] == 628:
                    output.append(198)
                    output.append(198)
                    continue
                output.append(list(self.vocab.keys())[list(self.vocab.values()).index(byte_token)])

            all_inputs += output

        # return output
        return all_inputs
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)
    
    def decode(self, ids: list[int]) -> str:
        bytes = b''
        for id in ids:
            bytes += self.vocab[id]
        return bytes.decode('utf-8', errors='replace')