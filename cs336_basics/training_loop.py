import argparse

parser = argparse.ArgumentParser(description="training parameters")

# model parameters
parser.add_argument("vocab_size", type=int, default=10000)
parser.add_argument("context_length", type=int, default=256)
parser.add_argument("d_model", type=int, default=512)
parser.add_argument("d_ff", type=int, default=1344)

parser.add_argument("rope_theta", type=int, default=10000)

# training parameters
parser.add_argument("learning_rate", type=float, default=1e-3)
parser.add_argument("learning_rate", type=float, default=1e-3)
parser.add_argument("batch_size", type=int, default=16)
parser.add_argument("beta1", type=float, default=0.9)
parser.add_argument("beta2", type=float, default=0.999)
parser.add_argument("epsilon", type=float, default=1e-8)
parser.add_argument("weight_decay", type=float, default=0.01)

# gradient clipping 
parser.add_argument("max_l2_norm", type=float, default=1e-2)

# learning rate schedule 
parser.add_argument("lr_min", type=float, default=1e-3)
parser.add_argument("lr_max", type=float, default=1)
parser.add_argument("its_warmup", type=int, default=100)
parser.add_argument("its_cooldown", type=int, default=1000)

args = parser.parse_args()

def train_model(dataset, model, iterations, save_dir, checkpoints=100):

    for it in range(iterations):
        print(f"Training iteration {it}...", end=' ', flush=True)

        inputs, targets = data_loading(dataset, args.batch_size, args.context_length, device)

        lr = learning_rate_schedule(it, args.lr_min, args.lr_max, args.its_warmup, args.its_cooldown)

        opt = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
        )

        opt.zero_grad()
        loss = cross_entropy(inputs, targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt.step()

        print(f"Loss {loss.cpu().item()}", flush=True)

        if it % checkpoints == 0:
            save_checkpoint(model, opt, it, f'save_dir/iteration{it}.pt')

def train(file):
    # load data as dataset  np.memmap 
    # set up model config 