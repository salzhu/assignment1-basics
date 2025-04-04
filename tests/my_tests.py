import numpy as np 
import regex as re 

from collections.abc import Iterable, Iterator
from test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return 
    
    def encode(self, text: str) -> list[int]:
        # preprocess 
        # for each thing, split into bytes 
        # merge in order 
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.finditer(PAT, text)

        all_inputs = []
        for pt in pre_tokens:
            all_inputs.append(list(pt.group().encode("utf-8"))) 

        for merge in self.merges: 
            for input_bytes in all_inputs:
                for i in range(len(input_bytes) - 2, -1, -1):
                    if input_bytes[i] == self.vocab[merge[0]] and input_bytes[i + 1] == self.vocab[merge[1]]:
                        input_bytes[i] = self.vocab[merge[0] + merge[1]]
                        input_bytes.pop(i + 1)
        
        return sum(all_inputs, [])
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)
    
    def decode(self, ids: list[int]) -> str:
        bytes = b''
        for id in ids:
            bytes += self.vocab[id]
        return bytes.decode('utf-8', errors='replace')
    
if __name__ == "__main__":
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string