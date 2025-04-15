import numpy as np 
import torch 
import pickle 
import argparse
import random

from transformer import softmax, TransformerLM
from tokenizer import BPETokenizer

parser = argparse.ArgumentParser()

# model parameters

parser.add_argument("--max_tokens", type=int, default=256)

parser.add_argument("--temp", type=float, default=0.2)

parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--n_layers", type=int, default=24)
parser.add_argument("--n_heads", type=int, default=8)

parser.add_argument("--prompt", type=str, default='This is a prompt')

parser.add_argument("--rope_theta", type=int, default=10000)

parser.add_argument("--save_dir", type=str, default='runs')
parser.add_argument("--model_name", type=str, default='lr_1e_3_32_0414')

args = parser.parse_args()

# i load the model 
# i take a prompt
# sample until hit end of text
# or max_tokens
# temperature

def generate(model, tokenizer, prompt, temperature=1.0, max_tokens=None, top_p=None, end_token='<|endoftext|>'):
    encoded_prompt = tokenizer.encode(prompt)
    encoded_special_token = tokenizer.encode(end_token)[0]

    token_list = encoded_prompt.copy()

    while max_tokens is None or (max_tokens is not None and len(token_list) - len(encoded_prompt) < max_tokens):
        # generate new token 
        # print(token_list)
        tokens = torch.Tensor([token_list[:model.context_length]]).int()
        # print(tokens)
        logits = model.forward(tokens)
        # testing end of text logits[0][-1][256] = 10
        probs = softmax(logits[0][-1], dim=0, temp=temperature)
        # if top_p is not None:
            # get top p probability tokens
        token = random.choices(np.arange(len(probs)), weights=probs)
        token_list.append(token[0])
        if token[0] == encoded_special_token:
            break

    return tokenizer.decode(token_list[len(encoded_prompt):])

if __name__ == '__main__':
    transformer = TransformerLM(
        args.vocab_size, 
        args.context_length, 
        args.d_model, 
        args.n_layers, 
        args.n_heads, 
        args.d_ff, 
        args.rope_theta
    )

    tokenizer = BPETokenizer({}, {}, ["<|endoftext|>"])
    tokenizer.from_files(
        'TinyStoriesV2-GPT4-train_v10000_vocab_0411.pickle',
        'TinyStoriesV2-GPT4-train_v10000_merges_0411.pickle', 
        ["<|endoftext|>"]
    )

    saved_state_dict = torch.load(f'{args.save_dir}/{args.model_name}/final.pt')
    saved_state_dict = saved_state_dict['model']

    # Create a new state dict with modified keys
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
        else:
            new_key = key
        new_state_dict[new_key] = value

    transformer.load_state_dict(new_state_dict)

    prompt = args.prompt

    print(generate(transformer, tokenizer, prompt, temperature=args.temp, max_tokens=args.max_tokens))

    