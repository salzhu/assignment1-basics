import numpy as np 
import regex as re 
from collections.abc import Iterable, Iterator
import heapq
import pickle
import os 
import io 
import multiprocessing
from tqdm import tqdm

def train_bpe_clean(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokens_to_index = {}
    pretokens_to_index_count = 0
    index_to_list_count = {}

    with open(input_path, 'r') as file:
        line = file.readline()
        curline = ''
        while line:
            
            first_index = len(line)
            first = 0
            for special_token in special_tokens: 
                index = line.find(special_token)
                if index != -1:
                    if index < first_index: 
                        first_index = index 
                        first = special_token 

            if index == -1:
                # keep updating 
                curline += line 
                line = file.readline()
                continue 

            # found a special token 
            curline += line[:first_index]
            # send curline to worker to pre-tokenize 
            pretoken_iterator = re.finditer(PAT, curline)
            for pt in pretoken_iterator: 
                if pt.group() not in pretokens_to_index: 
                    pretokens_to_index[pt.group()] = pretokens_to_index_count
                    index_to_list_count[pretokens_to_index_count] = [list(pt.group().encode("utf-8")), 0]
                    pretokens_to_index_count += 1
                    
                index_to_list_count[pretokens_to_index[pt.group()]][1] += 1
            curline = ''

            line = line[first_index + len(first):]
        pretoken_iterator = re.finditer(PAT, curline)
        for pt in pretoken_iterator: 
            if pt.group() not in pretokens_to_index: 
                pretokens_to_index[pt.group()] = pretokens_to_index_count
                index_to_list_count[pretokens_to_index_count] = [list(pt.group().encode("utf-8")), 0]
                pretokens_to_index_count += 1
                
            # print(pretokens_to_index)
            index_to_list_count[pretokens_to_index[pt.group()]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    for t in range(len(pretokens_to_index)):
        pretoken = index_to_list_count[t][0]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = []
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].append((t, s))
            pair_counts[(pretoken[s], pretoken[s + 1])] += index_to_list_count[t][1]

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}", end=' ', flush=True)

        for match in pairs[(l,r)]:
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 

            assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
            assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
            assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                continue
            
            if s > 0:
                left_token = index_to_list_count[t][0][s - 1]
                if (left_token, cur_vocab_size) not in pairs: # a l r
                    pairs[(left_token, cur_vocab_size)] = []
                    pair_counts[(left_token, cur_vocab_size)] = 0
                pairs[(left_token, cur_vocab_size)].append((t, s - len(vocab[left_token])))
                pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
            
            if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                if (cur_vocab_size, right_token) not in pairs:
                    pairs[(cur_vocab_size, right_token)] = []
                    pair_counts[(cur_vocab_size, right_token)] = 0
                pairs[(cur_vocab_size, right_token)].append((t, s))
                pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
            
            for j in range(len(vocab[l]) + len(vocab[r])):
                index_to_list_count[t][0][s + j] = cur_vocab_size 

        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1
        print('done', flush=True)

    return vocab, merges

def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]  # Chunks start on previous index, don't include last index
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            if mini_chunk == b"":  # If EOF, this boundary should be at the end of the file
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)  # Find the special token in the mini chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_worker(input_path, start, end, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        escaped_special_tokens = [re.escape(st) for st in special_tokens]
        pattern_special_tokens = "|".join(escaped_special_tokens)
        chunks = re.split(pattern_special_tokens, chunk)

    pretoken_counts = {}
    for cur_chunk in tqdm(chunks):
        pretoken_iterator = re.finditer(PAT, cur_chunk)
        for pt in pretoken_iterator: 
            token = pt.group()
            if token not in pretoken_counts:
                pretoken_counts[token] = 0
            pretoken_counts[token] += 1

    # return dicts and merge dicts outside of the function

    return pretoken_counts

def train_bpe_clean_splits(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    num_processes = multiprocessing.cpu_count()
    print(num_processes)

    full_pretoken_list = []

    all_chunks = []

    input_params = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # do this stuff in the other function 
            input_params.append((input_path, start, end, special_tokens))
        

    print(input_params)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Apply the function to all input lists in parallel
        pretoken_count_dicts = pool.starmap(pretokenize_worker, input_params)

    pretokens_to_index = {}
    pretokens_to_index_count = 0
    index_to_list_count = {}

    # merge the dicts
    for d in pretoken_count_dicts:
        for key, value in d.items():
            if key not in pretokens_to_index:
                pretokens_to_index[key] = pretokens_to_index_count
                index_to_list_count[pretokens_to_index_count] = [list(key.encode("utf-8")), 0]
                pretokens_to_index_count += 1
            index_to_list_count[pretokens_to_index[key]][1] += value

    # print(index_to_list_count)
    
    # full_pretoken_list = [item for sublist in pretokenized_chunks for item in sublist]
    # print(full_pretoken_list)

    # for pt in full_pretoken_list:
    #     if pt not in pretokens_to_index: 
    #         pretokens_to_index[pt] = pretokens_to_index_count
    #         index_to_list_count[pretokens_to_index_count] = [list(pt.encode("utf-8")), 0]
    #         pretokens_to_index_count += 1
            
    #     index_to_list_count[pretokens_to_index[pt]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    del pretoken_count_dicts

    for t in range(len(index_to_list_count)):
        pretoken = index_to_list_count[t][0]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = []
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].append((t, s))
            pair_counts[(pretoken[s], pretoken[s + 1])] += index_to_list_count[t][1]

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}", end=' ', flush=True)

        for match in pairs[(l,r)]:
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 

            assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
            assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
            assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                continue
            
            if s > 0:
                left_token = index_to_list_count[t][0][s - 1]
                if (left_token, cur_vocab_size) not in pairs: # a l r
                    pairs[(left_token, cur_vocab_size)] = []
                    pair_counts[(left_token, cur_vocab_size)] = 0
                pairs[(left_token, cur_vocab_size)].append((t, s - len(vocab[left_token])))
                pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
            
            if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                if (cur_vocab_size, right_token) not in pairs:
                    pairs[(cur_vocab_size, right_token)] = []
                    pair_counts[(cur_vocab_size, right_token)] = 0
                pairs[(cur_vocab_size, right_token)].append((t, s))
                pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
            
            for j in range(len(vocab[l]) + len(vocab[r])):
                index_to_list_count[t][0][s + j] = cur_vocab_size 

        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1
        print('done', flush=True)

    return vocab, merges, index_to_list_count, pretokens_to_index

def train_bpe_clean_splits_set(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    num_processes = multiprocessing.cpu_count()
    print(num_processes)

    input_params = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # do this stuff in the other function 
            input_params.append((input_path, start, end, special_tokens))
        

    print(input_params)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Apply the function to all input lists in parallel
        pretoken_count_dicts = pool.starmap(pretokenize_worker, input_params)

    pretokens_to_index = {}
    pretokens_to_index_count = 0
    index_to_list_count = {}

    # merge the dicts
    for d in pretoken_count_dicts:
        for key, value in d.items():
            if key not in pretokens_to_index:
                pretokens_to_index[key] = pretokens_to_index_count
                index_to_list_count[pretokens_to_index_count] = [list(key.encode("utf-8")), 0]
                pretokens_to_index_count += 1
            index_to_list_count[pretokens_to_index[key]][1] += value

    # print(index_to_list_count)
    
    # full_pretoken_list = [item for sublist in pretokenized_chunks for item in sublist]
    # print(full_pretoken_list)

    # for pt in full_pretoken_list:
    #     if pt not in pretokens_to_index: 
    #         pretokens_to_index[pt] = pretokens_to_index_count
    #         index_to_list_count[pretokens_to_index_count] = [list(pt.encode("utf-8")), 0]
    #         pretokens_to_index_count += 1
            
    #     index_to_list_count[pretokens_to_index[pt]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    del pretoken_count_dicts

    for t in range(len(index_to_list_count)):
        pretoken = index_to_list_count[t][0]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = set()
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].add((t, s, True))
            pair_counts[(pretoken[s], pretoken[s + 1])] += index_to_list_count[t][1]

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}", end=' ', flush=True)

        for match in pairs[(l,r)]:
            if match[2] == False:
                continue
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 

            # print(index_to_list_count[t][0][s])
            # print(index_to_list_count[t][0][s + len(vocab[l])])

            if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                continue

            assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
            assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
            assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            
            
            if s > 0:
                left_token = index_to_list_count[t][0][s - 1]
                if (left_token, cur_vocab_size) not in pairs: # a l r
                    pairs[(left_token, cur_vocab_size)] = set()
                    pair_counts[(left_token, cur_vocab_size)] = 0
                pairs[(left_token, cur_vocab_size)].add((t, s - len(vocab[left_token]), True))
                pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                if (t, s - len(vocab[left_token]), True) in pairs[(left_token, l)]:
                    pairs[(left_token, l)].remove((t, s - len(vocab[left_token]), True))
                    pairs[(left_token, l)].add((t, s - len(vocab[left_token]), False))
                # pairs[(left_token, l)][(t, s - len(vocab[left_token]))]
                pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
            
            if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                if (cur_vocab_size, right_token) not in pairs:
                    pairs[(cur_vocab_size, right_token)] = set()
                    pair_counts[(cur_vocab_size, right_token)] = 0
                pairs[(cur_vocab_size, right_token)].add((t, s, True))
                pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                if (t, s + len(vocab[l]), True) in pairs[(r, right_token)]:
                    pairs[(r, right_token)].remove((t, s + len(vocab[l]), True))
                    pairs[(r, right_token)].add((t, s + len(vocab[l]), False))
                pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
            
            for j in range(len(vocab[l]) + len(vocab[r])):
                index_to_list_count[t][0][s + j] = cur_vocab_size 

        pairs[(l, r)] = set()
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1
        print('done', flush=True)

    return vocab, merges, index_to_list_count, pretokens_to_index

def train_bpe_clean_splits_set_bad(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    num_processes = multiprocessing.cpu_count()
    print(num_processes)

    input_params = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # do this stuff in the other function 
            input_params.append((input_path, start, end, special_tokens))
        

    print(input_params)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Apply the function to all input lists in parallel
        pretoken_count_dicts = pool.starmap(pretokenize_worker, input_params)

    pretokens_to_index = {}
    pretokens_to_index_count = 0
    index_to_list_count = {}

    # merge the dicts
    for d in pretoken_count_dicts:
        for key, value in d.items():
            if key not in pretokens_to_index:
                pretokens_to_index[key] = pretokens_to_index_count
                index_to_list_count[pretokens_to_index_count] = [list(key.encode("utf-8")), 0]
                pretokens_to_index_count += 1
            index_to_list_count[pretokens_to_index[key]][1] += value

    # print(index_to_list_count)
    
    # full_pretoken_list = [item for sublist in pretokenized_chunks for item in sublist]
    # print(full_pretoken_list)

    # for pt in full_pretoken_list:
    #     if pt not in pretokens_to_index: 
    #         pretokens_to_index[pt] = pretokens_to_index_count
    #         index_to_list_count[pretokens_to_index_count] = [list(pt.encode("utf-8")), 0]
    #         pretokens_to_index_count += 1
            
    #     index_to_list_count[pretokens_to_index[pt]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    del pretokens_to_index, pretoken_count_dicts

    for t in range(len(index_to_list_count)):
        pretoken = index_to_list_count[t][0]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = set()
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].add((t, s))
            pair_counts[(pretoken[s], pretoken[s + 1])] += index_to_list_count[t][1]

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}", end=' ', flush=True)

        if l == r: 
            temp_list = list(pairs[(l,r)])
            for match in temp_list:
                # if match not in pairs[(l, r)]:
                #     continue
                t = match[0] 
                s = match[1] 
                vocab[cur_vocab_size] = vocab[l] + vocab[r] 

                assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
                assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
                assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
                if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                    continue
                
                if s > 0:
                    left_token = index_to_list_count[t][0][s - 1]
                    if (left_token, cur_vocab_size) not in pairs: # a l r
                        pairs[(left_token, cur_vocab_size)] = set()
                        pair_counts[(left_token, cur_vocab_size)] = 0
                    pairs[(left_token, cur_vocab_size)].add((t, s - len(vocab[left_token])))
                    if left_token == l and cur_vocab_size == r:
                        temp_list.append((t, s - len(vocab[left_token])))
                    pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                    pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                    if left_token == l and l == r:
                        temp_list.remove((t, s - len(vocab[left_token])))
                    pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
                
                if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                    right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                    if (cur_vocab_size, right_token) not in pairs:
                        pairs[(cur_vocab_size, right_token)] = set()
                        pair_counts[(cur_vocab_size, right_token)] = 0
                    pairs[(cur_vocab_size, right_token)].add((t, s))
                    if cur_vocab_size == l and right_token == r:
                        temp_list.append((t, s))
                    pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                    pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                    if r == l and right_token == r:
                        temp_list.remove((t, s + len(vocab[l])))
                    pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
                
                for j in range(len(vocab[l]) + len(vocab[r])):
                    index_to_list_count[t][0][s + j] = cur_vocab_size 

        else: 
            for match in pairs[(l,r)]:
                t = match[0] 
                s = match[1] 
                vocab[cur_vocab_size] = vocab[l] + vocab[r] 

                assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
                assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
                assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
                if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                    continue
                
                
                if s > 0:
                    left_token = index_to_list_count[t][0][s - 1]
                    if (left_token, cur_vocab_size) not in pairs: # a l r
                        pairs[(left_token, cur_vocab_size)] = set()
                        pair_counts[(left_token, cur_vocab_size)] = 0
                    pairs[(left_token, cur_vocab_size)].add((t, s - len(vocab[left_token])))
                    pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                    pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                    pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
                
                if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                    right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                    if (cur_vocab_size, right_token) not in pairs:
                        pairs[(cur_vocab_size, right_token)] = set()
                        pair_counts[(cur_vocab_size, right_token)] = 0
                    pairs[(cur_vocab_size, right_token)].add((t, s))
                    pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                    pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                    pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
                
                for j in range(len(vocab[l]) + len(vocab[r])):
                    index_to_list_count[t][0][s + j] = cur_vocab_size 

        pairs[(l, r)] = set()
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1
        print('done', flush=True)

    return vocab, merges

class BPETokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.index = None
        self.pretokens = None
    
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as file:
            self.vocab = pickle.load(file) 
        """
        To save the vocab
        with open('vocab_path.pickle', 'wb') as file:
            pickle.dump(vocab, file)
        """

        with open(merges_filepath, 'rb') as file:
            self.merges = pickle.load(file) 
        """
        To save the merges
        with open('merges_path.pickle', 'wb') as file:
            pickle.dump(merges, file)
        """

        self.special_tokens = special_tokens

    def pretokenize_parallel(self, input_path, start, end, special_tokens, special_token_id):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            escaped_special_tokens = [re.escape(st) for st in special_tokens]
            pattern_special_tokens = "|".join(escaped_special_tokens)
            chunks = re.split(pattern_special_tokens, chunk)

        pretoken_list = []
        out = []
        for cur_chunk in tqdm(chunks):
            pretoken_iterator = re.finditer(PAT, cur_chunk)
            for pt in pretoken_iterator: 
                pretoken = pt.group()
                # pretoken_list.append(token)
                
                if pretoken in special_tokens:
                    # out.append(list(self.vocab.keys())[vocab_list.index(pretoken)])
                    out += [special_token_id]
                elif pretoken in self.pretokens:
                    # pretoken list is self.index[self.pretokens[pretoken]]
                    # len(bytes(vocab[439]))
                    token_list = self.index[self.pretokens[pretoken]][0]
                    i = 0 
                    while i < len(token_list):
                        out += [token_list[i]]
                        i += len(bytes(self.vocab[token_list[i]]))
                        # i += 1
                else:
                    # print(pretoken)
                    out += self.encode(pretoken)
            # pretoken_list.append(special_tokens[0])
            out += [special_token_id]

        return out

        # return dicts and merge dicts outside of the function
        print(pretoken_list)

        return pretoken_list

    def encode_from_pretokens(self, input_path : str, vocab_filepath, merges_filepath, index_filepath, pretoken_index_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as file:
            self.vocab = pickle.load(file) 

        with open(merges_filepath, 'rb') as file:
            self.merges = pickle.load(file) 

        with open(index_filepath, 'rb') as file:
            self.index = pickle.load(file) 

        with open(pretoken_index_filepath, 'rb') as file:
            self.pretokens = pickle.load(file) 

        num_processes = multiprocessing.cpu_count()
        num_processes = 4
        print(num_processes)

        input_params = []

        vocab_list = list(self.vocab.values())
        special_token_id = list(self.vocab.keys())[vocab_list.index(special_tokens[0].encode("utf-8"))]

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
            # The following is a serial implementation, but you can parallelize this by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                # do this stuff in the other function 
                input_params.append((input_path, start, end, special_tokens, special_token_id))
            
        print(input_params)

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Apply the function to all input lists in parallel
            pretoken_lists = pool.starmap(self.pretokenize_parallel, input_params)

        print('done')

        pretokens = []
        for temp in pretoken_lists:
            pretokens += temp
        if pretokens[-1] == special_token_id:
            pretokens = pretokens[:-1]

        # i want the list of pretokens in order
        # now, i have pretokens_to_index and index_to_list 
        # assume pretokens is list of pretokens in order 

        # print(pretokens)

        # out = []
        # vocab_list = list(self.vocab.values())
        # special_token_id = list(self.vocab.keys())[vocab_list.index(special_tokens[0].encode("utf-8"))]
        # for pretoken in tqdm(pretokens):
        #     if pretoken in special_tokens:
        #         # out.append(list(self.vocab.keys())[vocab_list.index(pretoken)])
        #         out += [special_token_id]
        #     elif pretoken in self.pretokens:
        #         # pretoken list is self.index[self.pretokens[pretoken]]
        #         # len(bytes(vocab[439]))
        #         token_list = self.index[self.pretokens[pretoken]][0]
        #         i = 0 
        #         while i < len(token_list):
        #             out += [token_list[i]]
        #             i += len(bytes(self.vocab[token_list[i]]))
        #             # i += 1
        #     else:
        #         # print(pretoken)
        #         out += self.encode(pretoken)

        # print('<|endoftext|>' in self.vocab)
        # print(self.vocab[256])
        return pretokens
    
    def encode(self, text: str) -> list[int]:

        # print(('\n'.encode("utf-8"), '\n'.encode("utf-8")) in self.merges)

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        # print(self.special_tokens)

        separated = [text]
        for st in self.special_tokens:
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
            # print(list(pre_tokens))

            byte_tokens = []

            for pt in pre_tokens: 
                int_tokens = list(pt.group().encode("utf-8"))
                temp = []
                for i in range(len(int_tokens)):
                    t = int_tokens[i]
                    temp.append(bytes([t]))
                byte_tokens.append(temp)

            for merge in self.merges: 
                for byte_token in byte_tokens:
                    for i in range(len(byte_token) - 2, -1, -1):
                        if byte_token[i] == merge[0] and byte_token[i + 1] == merge[1]:
                            byte_token[i] = merge[0] + merge[1]
                            byte_token.pop(i + 1)

            output = [] 
            for byte_token in byte_tokens:
                for tok in byte_token:
                    output.append(list(self.vocab.keys())[list(self.vocab.values()).index(tok)])

            all_inputs += output

        return all_inputs
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)
    
    def decode(self, ids: list[int]) -> str:
        bytes = b''
        for id in ids:
            bytes += self.vocab[id]
        return bytes.decode('utf-8', errors='replace')
    

if __name__ == '__main__':
    vocab, merges = train_bpe_clean_splits('/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt', 1000, ['<|endoftext|>'])
    # print(merges)