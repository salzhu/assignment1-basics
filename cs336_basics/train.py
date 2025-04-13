import numpy as np 
import torch 
import torch.nn as nn
from einops import rearrange, einsum

"""Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
def cross_entropy(inputs, targets):
    # print(inputs.shape)
    # print(torch.min(inputs))
    # print(inputs)
    loss = torch.tensor(0.0) 
    inputs -= torch.unsqueeze(torch.max(inputs, dim=1).values,1)
    # print(torch.min(inputs, dim=1))
    # inputs -= torch.min(inputs)
    # print(inputs)
    for i in range(len(inputs)):
        loss -= inputs[i][targets[i]]
        loss += torch.log(torch.sum(torch.exp(inputs[i][:])))
    return loss / len(inputs)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # print(inputs.shape)
        # print(torch.min(inputs))
        # print(inputs)
        loss = torch.tensor(0.0, device=inputs.device) 
        inputs -= torch.unsqueeze(torch.max(inputs, dim=1).values,1)
        # print(torch.min(inputs, dim=1))
        # inputs -= torch.min(inputs)
        # print(inputs)
        print(inputs.shape)
        for i in range(len(inputs)):
            loss -= inputs[i][targets[i]]
            loss += torch.log(torch.sum(torch.exp(inputs[i][:])))
        return loss / len(inputs)

def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return t / T_w * a_max 
    elif t <= T_c:
        return a_min + 0.5 * (1 + np.cos((t - T_w) / (T_c - T_w) * np.pi)) * (a_max - a_min)
    else:
        return a_min 
    
def gradient_clipping(params, max_l2_norm, eps=1e-6):
    grads = []
    for param in params:
        if param.grad is not None:
            # print(param.grad.shape)
            grads.append(param.grad.view(-1))

    all_grads = torch.cat(grads, dim=0)

    norm = torch.norm(all_grads)

    if norm >= max_l2_norm:
        for param in params:
            if param.grad is not None:
                param.grad *= max_l2_norm / (norm + eps)
            # norm = torch.norm(param.grad)
            # print(norm)

def data_loading(dataset, batch_size, context_length, device):
    # temp = np.memmap(dataset, dtype='int', mode='r')
    # print(temp.shape)
    inputs = []
    targets = []
    possible_start = np.arange(0, len(dataset) - context_length)
    for i in range(batch_size):
        ind = np.random.choice(possible_start) 
        inputs.append(dataset[ind:ind + context_length])
        targets.append(dataset[ind + 1:ind + 1 + context_length])
    # inputs = dataset[:-1]
    # targets = dataset[1:]
    # print(torch.tensor(dataset))
    # print(batch_size, context_length)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))
    return 

def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration 
    torch.save(obj, out)
    return 

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']