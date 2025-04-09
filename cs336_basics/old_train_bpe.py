
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

        # print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {pairs[l, r]}")
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

def train_bpe_new2(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    pretokens = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokens_dict = {}

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
                pretokens.append(list(pt.group().encode("utf-8")))
                # if pt.group() not in pretokens_dict: 
                #     pretokens_dict[pt.group()] = 0
                # pretokens_dict[pt.group()] += 1
            curline = ''

            line = line[first_index + len(first):]
        pretoken_iterator = re.finditer(PAT, curline)
        for pt in pretoken_iterator: 
            pretokens.append(list(pt.group().encode("utf-8")))
            # if pt.group() not in pretokens_dict: 
            #     pretokens_dict[pt.group()] = 0
            #     pretokens_dict[pt.group()] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    for t in range(len(pretokens)):
        pretoken = pretokens[t]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = []
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].append((t, s))
            pair_counts[(pretoken[s], pretoken[s + 1])] += 1

    # for pretoken in list(pretokens_dict.keys()):
    #     for s in range(len(pretoken) - 1):
    #         if (pretoken[s], pretoken[s + 1]) not in pairs: 
    #             pairs[(pretoken[s], pretoken[s + 1])] = []
    #             pair_counts[(pretoken[s], pretoken[s + 1])] = 0
    #         pairs[(pretoken[s], pretoken[s + 1])].append((t, s))
    #         pair_counts[(pretoken[s], pretoken[s + 1])] += 1

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    # print(pairs)
    # print(pair_counts)
    # print("pairs printed!")

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}")
        # print(pairs[(l,r)])

        for match in pairs[(l,r)]:
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 
            
            # print(l)
            # print(pretokens[t][s], pretokens[t][s + len(vocab[l])])
            assert pretokens[t][s] == l and pretokens[t][s + len(vocab[l])] == r, "left and right don't match in match"
            assert pretokens[t][s + len(vocab[l]) - 1] == l
            assert pretokens[t][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            if pretokens[t][s] == cur_vocab_size: # bad merge
                continue
            # print(t, s)
            if s > 0:
                if (pretokens[t][s - 1], cur_vocab_size) not in pairs: # a l r
                    pairs[(pretokens[t][s - 1], cur_vocab_size)] = []
                    pair_counts[(pretokens[t][s - 1], cur_vocab_size)] = 0
                # print(pretokens[t][s - 1], cur_vocab_size)
                pairs[(pretokens[t][s - 1], cur_vocab_size)].append((t, s - len(vocab[pretokens[t][s - 1]])))
                pair_counts[(pretokens[t][s - 1], cur_vocab_size)] += 1
                pairs[(pretokens[t][s - 1], l)].remove((t, s - len(vocab[pretokens[t][s - 1]])))
                pair_counts[(pretokens[t][s - 1], l)] -= 1
            if s < len(pretokens[t]) - len(vocab[l]) - len(vocab[r]): # l r a
                if (cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])]) not in pairs:
                    pairs[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] = []
                    pair_counts[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] = 0
                pairs[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])].append((t, s))
                pair_counts[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] += 1
                # print((l, pretokens[t][s + len(vocab[l])], pretokens[t][s + len(vocab[l]) + len(vocab[r])]))
                pairs[(r, pretokens[t][s + len(vocab[l]) + len(vocab[r])])].remove((t, s + len(vocab[l])))
                pair_counts[(r, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] -= 1
            for j in range(len(vocab[l]) + len(vocab[r])):
                pretokens[t][s + j] = cur_vocab_size 
            # print(pairs[(l,r)])
            # remove things 
            # for new_match in pairs[(l, r)]: 
            #     if new_match[0] == t and new_match[1] < s + len(vocab[l]) + len(vocab[r]):
            #         pairs[(l, r)].remove(new_match)
            #         pair_counts[(l, r)] -= 1
            # print(pairs)
        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1

    return vocab, merges

def train_bpe_new2_set(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    pretokens = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
                pretokens.append(list(pt.group().encode("utf-8")))
                # if pt.group() not in pretokens_dict: 
                #     pretokens_dict[pt.group()] = 0
                # pretokens_dict[pt.group()] += 1
            curline = ''

            line = line[first_index + len(first):]
        pretoken_iterator = re.finditer(PAT, curline)
        for pt in pretoken_iterator: 
            pretokens.append(list(pt.group().encode("utf-8")))
            # if pt.group() not in pretokens_dict: 
            #     pretokens_dict[pt.group()] = 0
            #     pretokens_dict[pt.group()] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    for t in range(len(pretokens)):
        pretoken = pretokens[t]
        for s in range(len(pretoken) - 1):
            if (pretoken[s], pretoken[s + 1]) not in pairs: 
                pairs[(pretoken[s], pretoken[s + 1])] = set()
                pair_counts[(pretoken[s], pretoken[s + 1])] = 0
            pairs[(pretoken[s], pretoken[s + 1])].add((t, s))
            pair_counts[(pretoken[s], pretoken[s + 1])] += 1

    # for pretoken in list(pretokens_dict.keys()):
    #     for s in range(len(pretoken) - 1):
    #         if (pretoken[s], pretoken[s + 1]) not in pairs: 
    #             pairs[(pretoken[s], pretoken[s + 1])] = []
    #             pair_counts[(pretoken[s], pretoken[s + 1])] = 0
    #         pairs[(pretoken[s], pretoken[s + 1])].append((t, s))
    #         pair_counts[(pretoken[s], pretoken[s + 1])] += 1

    # for i in range(cur_vocab_size):
    #     for j in range(cur_vocab_size):
    #         # pair_counts.append((-len(pairs[i][j]), (i, j)))
    # heapq.heapify(pair_counts)

    # getting the max (l, r) pair: heapq.heappop(pair_counts)[1]
    merges = []

    # print(pairs)
    # print(pair_counts)
    # print("pairs printed!")

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}")
        # print(pairs[(l,r)])

        for match in pairs[(l,r)]:
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 
            
            # print(l)
            # print(pretokens[t][s], pretokens[t][s + len(vocab[l])])
            assert pretokens[t][s] == l and pretokens[t][s + len(vocab[l])] == r, "left and right don't match in match"
            assert pretokens[t][s + len(vocab[l]) - 1] == l
            assert pretokens[t][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            if pretokens[t][s] == cur_vocab_size: # bad merge
                continue
            # print(t, s)
            if s > 0:
                if (pretokens[t][s - 1], cur_vocab_size) not in pairs: # a l r
                    pairs[(pretokens[t][s - 1], cur_vocab_size)] = set()
                    pair_counts[(pretokens[t][s - 1], cur_vocab_size)] = 0
                # print(pretokens[t][s - 1], cur_vocab_size)
                pairs[(pretokens[t][s - 1], cur_vocab_size)].add((t, s - len(vocab[pretokens[t][s - 1]])))
                pair_counts[(pretokens[t][s - 1], cur_vocab_size)] += 1
                if not (pretokens[t][s - 1] == l and l == r):
                    pairs[(pretokens[t][s - 1], l)].remove((t, s - len(vocab[pretokens[t][s - 1]])))
                    pair_counts[(pretokens[t][s - 1], l)] -= 1
            if s < len(pretokens[t]) - len(vocab[l]) - len(vocab[r]): # l r a
                if (cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])]) not in pairs:
                    pairs[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] = set()
                    pair_counts[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] = 0
                pairs[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])].add((t, s))
                pair_counts[(cur_vocab_size, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] += 1
                # print((l, pretokens[t][s + len(vocab[l])], pretokens[t][s + len(vocab[l]) + len(vocab[r])]))
                if not (r == l and pretokens[t][s + len(vocab[l]) + len(vocab[r])] == r): 
                    pairs[(r, pretokens[t][s + len(vocab[l]) + len(vocab[r])])].remove((t, s + len(vocab[l])))
                    pair_counts[(r, pretokens[t][s + len(vocab[l]) + len(vocab[r])])] -= 1
            for j in range(len(vocab[l]) + len(vocab[r])):
                pretokens[t][s + j] = cur_vocab_size 
            # print(pairs[(l,r)])
            # remove things 
            # for new_match in pairs[(l, r)]: 
            #     if new_match[0] == t and new_match[1] < s + len(vocab[l]) + len(vocab[r]):
            #         pairs[(l, r)].remove(new_match)
            #         pair_counts[(l, r)] -= 1
            # print(pairs)
        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1

    return vocab, merges

def train_bpe_new3(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # build initial vocabulary 
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    cur_vocab_size = 256 + len(special_tokens)
    
    # read the file line by line, splitting by special tokens

    # pretokens = []
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
                # pretokens.append(list(pt.group().encode("utf-8")))
                # print(pt.group())
                if pt.group() not in pretokens_to_index: 
                    pretokens_to_index[pt.group()] = pretokens_to_index_count
                    index_to_list_count[pretokens_to_index_count] = [list(pt.group().encode("utf-8")), 0]
                    pretokens_to_index_count += 1
                    
                # print(pretokens_to_index)
                index_to_list_count[pretokens_to_index[pt.group()]][1] += 1
            curline = ''

            line = line[first_index + len(first):]
        pretoken_iterator = re.finditer(PAT, curline)
        for pt in pretoken_iterator: 
            # pretokens.append(list(pt.group().encode("utf-8")))
            # print(pt.group())
            if pt.group() not in pretokens_to_index: 
                pretokens_to_index[pt.group()] = pretokens_to_index_count
                index_to_list_count[pretokens_to_index_count] = [list(pt.group().encode("utf-8")), 0]
                pretokens_to_index_count += 1
                
            # print(pretokens_to_index)
            index_to_list_count[pretokens_to_index[pt.group()]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    # print(index_to_list_count)
    # print(pretokens_to_index)

    # pretokens = list(pretokens_dict.keys())

    # for pretoken in list(pretokens_dict.keys()):
    for t in range(len(pretokens_to_index)):
        # pretoken = list(pretokens[t].encode("utf-8")) # pretokens[t] 
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

    # print(pairs)
    # print(pair_counts)
    # print("pairs printed!")

    # print(pretokens_dict)
    # print(pair_counts)
    # print(pairs)

    while cur_vocab_size < vocab_size:
        # max_pair = heapq.heappop(pair_counts)[1]

        max_len = max(pair_counts.values())
        # print(vocab)
        # print(pair_counts)
        max_keys = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_len]
        max_pair = max(max_keys)
        l = list(vocab.keys())[list(vocab.values()).index(max_pair[0])]
        r = list(vocab.keys())[list(vocab.values()).index(max_pair[1])]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}")
        # print(pairs[(l,r)])

        for match in pairs[(l,r)]:
            t = match[0] 
            s = match[1] 
            vocab[cur_vocab_size] = vocab[l] + vocab[r] 
            # print(index_to_list_count[t])
            
            # print(l)
            # print(pretokens[t][s], pretokens[t][s + len(vocab[l])])
            # print(index_to_list_count[t][0])
            # print(t, s)
            # print(index_to_list_count[t])
            assert index_to_list_count[t][0][s] == l and index_to_list_count[t][0][s + len(vocab[l])] == r, "left and right don't match in match"
            assert index_to_list_count[t][0][s + len(vocab[l]) - 1] == l
            assert index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r]) - 1] == r
            if index_to_list_count[t][0][s] == cur_vocab_size: # bad merge
                continue
            
            # print(t, s)
            if s > 0:
                if (index_to_list_count[t][0][s - 1], cur_vocab_size) not in pairs: # a l r
                    pairs[(index_to_list_count[t][0][s - 1], cur_vocab_size)] = []
                    pair_counts[(index_to_list_count[t][0][s - 1], cur_vocab_size)] = 0
                # print(pretokens[t][s - 1], cur_vocab_size)
                pairs[(index_to_list_count[t][0][s - 1], cur_vocab_size)].append((t, s - len(vocab[index_to_list_count[t][0][s - 1]])))
                pair_counts[(index_to_list_count[t][0][s - 1], cur_vocab_size)] += index_to_list_count[t][1] # 1
                pairs[(index_to_list_count[t][0][s - 1], l)].remove((t, s - len(vocab[index_to_list_count[t][0][s - 1]])))
                pair_counts[(index_to_list_count[t][0][s - 1], l)] -= index_to_list_count[t][1] # 1
            if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                if (cur_vocab_size, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]) not in pairs:
                    pairs[(cur_vocab_size, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])] = []
                    pair_counts[(cur_vocab_size, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])] = 0
                pairs[(cur_vocab_size, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])].append((t, s))
                pair_counts[(cur_vocab_size, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])] += index_to_list_count[t][1] # 1
                # print((l, pretokens[t][s + len(vocab[l])], pretokens[t][s + len(vocab[l]) + len(vocab[r])]))
                # print(r, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])
                pairs[(r, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])].remove((t, s + len(vocab[l])))
                pair_counts[(r, index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])])] -= index_to_list_count[t][1] # 1
            for j in range(len(vocab[l]) + len(vocab[r])):
                index_to_list_count[t][0][s + j] = cur_vocab_size 
            # print(pairs[(l,r)])
            # remove things 
            # for new_match in pairs[(l, r)]: 
            #     if new_match[0] == t and new_match[1] < s + len(vocab[l]) + len(vocab[r]):
            #         pairs[(l, r)].remove(new_match)
            #         pair_counts[(l, r)] -= 1
            # print(pairs)
        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1

    return vocab, merges

def train_bpe_new3_set(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
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
                
            index_to_list_count[pretokens_to_index[pt.group()]][1] += 1

    # building original pairs with sliding window 
    pair_counts = {}
    pairs = {}

    for t in range(len(pretokens_to_index)):
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

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}")

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
                pairs[(left_token, cur_vocab_size)].add((t, s - len(left_token)))
                pair_counts[(left_token, cur_vocab_size)] += index_to_list_count[t][1] # 1
                pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                pair_counts[(left_token, l)] -= index_to_list_count[t][1] # 1
                if not (left_token == l and l == r):
                    pairs[(left_token, l)].remove((t, s - len(vocab[left_token])))
                    pair_counts[(left_token, l)] -= index_to_list_count[t][1]
            
            if s < len(index_to_list_count[t][0]) - len(vocab[l]) - len(vocab[r]): # l r a
                right_token = index_to_list_count[t][0][s + len(vocab[l]) + len(vocab[r])]
                if (cur_vocab_size, right_token) not in pairs:
                    pairs[(cur_vocab_size, right_token)] = set()
                    pair_counts[(cur_vocab_size, right_token)] = 0
                pairs[(cur_vocab_size, right_token)].add((t, s))
                pair_counts[(cur_vocab_size, right_token)] += index_to_list_count[t][1] # 1
                pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                pair_counts[(r, right_token)] -= index_to_list_count[t][1] # 1
                if not (r == l and right_token == r): 
                    pairs[(r, right_token)].remove((t, s + len(vocab[l])))
                    pair_counts[(r, right_token)] -= index_to_list_count[t][1]
            
            for j in range(len(vocab[l]) + len(vocab[r])):
                index_to_list_count[t][0][s + j] = cur_vocab_size 

        pairs[(l, r)] = []
        pair_counts[(l, r)] = 0

        cur_vocab_size += 1

    return vocab, merges

def train_bpe_new(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    cur_vocab_size = 256
    merges = []

    # pairs = np.full((vocab_size, vocab_size), [], dtype=object)

    pairs = [[] for _ in range(vocab_size)]  # Create rows
    for i in range(vocab_size):
        pairs[i] = [[] for _ in range(vocab_size)]

    file = open(input_path, 'r')
    file_content = file.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.finditer(PAT, file_content)

    # cur_index = 0

    # build the pairs lists with indexes

    all_tokens = []
    for pt in pre_tokens:
        all_tokens.append(list(pt.group().encode("utf-8")))

    j = 0 # pt count

    for byte_list in all_tokens:
        # pt = pre_tokens[j]
        # byte_list = list(pt.group().encode("utf-8"))
        # print(len(byte_list))
        for i in range(len(byte_list) - 1):
            # pair = (byte_list[i], byte_list[i + 1])
            pairs[byte_list[i]][byte_list[i + 1]].append((j, i))
            # pairs[pair].append((j, i)) # the pre-token and position in the pre-token
            # cur_index += 1

        j += 1

    print(pairs[:100])

    while cur_vocab_size < vocab_size - len(special_tokens): 
        # find most common pair 
        # max_pairs = np.argwhere(pairs==pairs.max())
        max_pairs = [] 
        max_len = 0

        for p1 in range(len(pairs)):
            for p2 in range(len(pairs)):
                if len(pairs[p1][p2]) > max_len: 
                    max_len = len(pairs[p1][p2]) 
                    max_pairs = [(p1, p2)]
                elif len(pairs[p1][p2]) == max_len:
                    max_pairs.append((p1, p2))

        # print(max_pairs)

        # max_pairs_alph = np.argmax([(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs])
        # print([(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs])
        # print([(vocab[l], vocab[r]) for (l, r) in max_pairs])
        pairs_encoded = [(vocab[l].decode("utf-8"), vocab[r].decode("utf-8")) for (l, r) in max_pairs]
        # print(pairs_encoded)
        max_pair = max(pairs_encoded)
        # if (' .','..') in pairs_encoded:
        #     max_pair = (' .','..')
        l = 0
        r = 0 
        for pair in max_pairs:
            if (vocab[pair[0]].decode("utf-8"), vocab[pair[1]].decode("utf-8")) == max_pair:
                l, r = pair
                break

        # , key = lambda k: square[k]

        merges.append((vocab[l], vocab[r]))

        print(f"merging {vocab[l]} and {vocab[r]} into {cur_vocab_size} with count {max_len}")

        for (j, i) in pairs[l][r]:
            if i > 0: 
                # merging left token a l r --> a lr (lr)
                pairs[all_tokens[j][i - 1]][cur_vocab_size].append((j, i - 1))
                print(all_tokens[j][i-1], l)
                pairs[all_tokens[j][i - 1]][l].remove((j, i - 1))
                # pairs[l][r].remove((j, i))
                all_tokens[j][i] = cur_vocab_size 
                all_tokens[j][i + 1] = cur_vocab_size 
            if i < len(all_tokens[j]) - 2:
                # merging right token l r b --> lr (lr) b
                # print(all_tokens[j][i + 2])
                len_merged = len(vocab[l] + vocab[r])
                pairs[cur_vocab_size][all_tokens[j][i + len_merged]].append((j, i)) # always append the first 
                pairs[r][all_tokens[j][i + 2]].remove((j, i + 1))
                # pairs[l][r].remove((j, i))
                all_tokens[j][i] = cur_vocab_size 
                all_tokens[j][i + 1] = cur_vocab_size 

        pairs[l][r] = []

        print(f"done merging count {len(pairs[l][r])}")


        # print(pairs[l][r])

        # cur_j = 0 
        # #print(len(pre_tokens))
        # pre_tokens = re.finditer(PAT, file_content)

        # for pt in pre_tokens: 
        #     print('here')
        #     if len(pairs[l][r]) == 0:
        #         break
        #     (new_j, i) = pairs[l][r][0]
        #     if new_j != cur_j: 
        #         print('here')
        #         cur_j += 1 
        #         continue 
        #     if i > 0: 
        #         # merging left token a l r --> a lr
        #         pairs[pt[i - 1]][cur_vocab_size].append((cur_j, i - 1))
        #         pairs[pt[i - 1]][l].remove((cur_j, i - 1))
        #     if i < len(pt) - 1:
        #         # merging right token l r b --> lr b
        #         pairs[cur_vocab_size][pt[i + 2]].append((cur_j, i))
        #         pairs[r][pt[i + 2]].remove((cur_j, i + 1))
        #     pairs[l][r] = pairs[l][r][1:]
        #     print('here')

        #     cur_j += 1

        #print(pairs[l][r])


        # for (j, i) in pairs[l][r]:
        #     if i > 0: 
        #         # merging left token a l r --> a lr
        #         pairs[pre_tokens[j][i - 1]][cur_vocab_size].append((j, i - 1))
        #         pairs[pre_tokens[j][i - 1]][l].remove((j, i - 1))
        #     if i < len(pre_tokens[j]) - 1:
        #         # merging right token l r b --> lr b
        #         pairs[cur_vocab_size][pre_tokens[j][i + 2]].append((j, i))
        #         pairs[r][pre_tokens[j][i + 2]].remove((j, i + 1))

        # pairs[l, r] = []

        # # replace everything in input_bytes
        # for input_bytes in all_inputs:
        #     for i in range(len(input_bytes) - 2, -1, -1):
        #         if input_bytes[i] == l and input_bytes[i + 1] == r:
        #             if i > 0:
        #                 pairs[input_bytes[i - 1], l] -= 1
        #                 pairs[input_bytes[i - 1], cur_vocab_size] += 1
        #             if i < len(input_bytes) - 2:
        #                 pairs[r, input_bytes[i + 2]] -= 1
        #                 pairs[cur_vocab_size, input_bytes[i + 2]] += 1
        #             input_bytes[i] = cur_vocab_size
        #             input_bytes.pop(i + 1)

        vocab[cur_vocab_size] = vocab[l] + vocab[r] 
        cur_vocab_size += 1
    
    for special_token in special_tokens:
        vocab[cur_vocab_size] = special_token.encode("utf-8")
        cur_vocab_size += 1

    return vocab, merges
