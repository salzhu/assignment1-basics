import argparse
import torch
import wandb
import os
import numpy as np 
from tqdm import tqdm

from datetime import datetime

torch.autograd.set_detect_anomaly(True)

from tokenizer import BPETokenizer
from train import learning_rate_schedule, gradient_clipping, save_checkpoint, load_checkpoint, CrossEntropyLoss
from train import load_batch
from transformer import TransformerLM
from optimizer import AdamW

parser = argparse.ArgumentParser()

# loading tokenizer
parser.add_argument("--train", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt')
parser.add_argument("--valid", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt')

parser.add_argument("--text_path", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt')
parser.add_argument("--vocab_path", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_vocab_0411.pickle')
parser.add_argument("--merges_path", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_merges_0411.pickle')
parser.add_argument("--index_path", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_index_to_list_0411.pickle')
parser.add_argument("--pretokens_path", type=str, default='/Users/sallyzhu/Desktop/cs336/assignment1-basics/TinyStoriesV2-GPT4-train_v10000_pretoken_index_0411.pickle')

# model parameters
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_heads", type=int, default=16)

parser.add_argument("--rope_theta", type=int, default=10000)

# training parameters
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.01)

# gradient clipping 
parser.add_argument("--max_l2_norm", type=float, default=1e-2)

# learning rate schedule 
parser.add_argument("--lr_min", type=float, default=1e-3)
parser.add_argument("--lr_max", type=float, default=1)
parser.add_argument("--its", type=int, default=1000)
parser.add_argument("--its_warmup", type=int, default=100)
parser.add_argument("--its_cooldown", type=int, default=800)

# saving
parser.add_argument("--save_dir", type=str, default='runs')
parser.add_argument("--wandb_name", type=str, default='temp')
parser.add_argument("--model_name", type=str, default='temp')

args = parser.parse_args()

torch.set_float32_matmul_precision('high')

def train_model(dataset, val_set, model, iterations, save_dir, model_name, checkpoints=10000):

    model = torch.compile(model)

    wandb.init(project=f"{args.wandb_name}", name=args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"on device {device}")

    loss_fn = CrossEntropyLoss()

    opt = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
    )

    inputs, targets = load_batch(dataset, args.batch_size, args.context_length, device)

    best_val_loss = 1000

    for it in range(iterations):
        # now = datetime.now()
        print(f"Training iteration {it}...", end=' ', flush=True)

        # inputs, targets = load_batch(dataset, args.batch_size, args.context_length, device)
        lr = learning_rate_schedule(it, 
                                    args.learning_rate, 
                                    args.lr_max, 
                                    0.1 * args.its, 
                                    args.its)

        opt.defaults['lr'] = lr

        opt.zero_grad()
        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = loss_fn(outputs, targets)

        loss.backward()
        del inputs, targets
        inputs, targets = load_batch(dataset, args.batch_size, args.context_length, device)

        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt.step()

        print(f"Loss {loss.cpu().item()}", flush=True)
        wandb.log({"train_loss": loss.cpu().item()}, step=it)

        if it % 100 == 0: # compute validation loss

            val_inputs, val_targets = load_batch(val_set, args.batch_size, args.context_length, device)
            
            val_outputs = model(val_inputs)
            val_outputs = val_outputs.view(-1, val_outputs.size(-1))
            val_targets = val_targets.view(-1)
            
            with torch.no_grad():
                val_loss = loss_fn(val_outputs, val_targets)
            print(f"Val. Loss {val_loss.cpu().item()}", flush=True)
            wandb.log({"val_loss": val_loss.cpu().item()}, step=it)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_checkpoint(model, opt, it, f'{save_dir}/{model_name}/best.pt')

            del val_outputs, val_loss

        if it % checkpoints == 0:
            save_checkpoint(model, opt, it, f'{save_dir}/{model_name}/iteration{it}.pt')

        if it == iterations - 1: # final model
            save_checkpoint(model, opt, it, f'{save_dir}/{model_name}/final.pt')

        del loss, outputs
        # del inputs, targets
        # print(datetime.now() - now)
        # del opt

    print("Done training!")
    wandb.finish()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_encoded = np.lib.format.open_memmap(args.train, mode='r').astype(int)
    valid_encoded = np.lib.format.open_memmap(args.valid, mode='r').astype(int)
    
    # set up model config 
    transformer = TransformerLM(
        args.vocab_size, 
        args.context_length, 
        args.d_model, 
        args.n_layers, 
        args.n_heads, 
        args.d_ff, 
        args.rope_theta, 
        device=device
    )

    # make dir {save_dir}/{model_name}
    os.makedirs(f'{args.save_dir}/{args.model_name}', exist_ok=True)

    train_model(train_encoded, valid_encoded, transformer, args.its, args.save_dir, args.model_name)

if __name__ == '__main__':
    train()