import os
import torch
from model import GPTConfig, GPT
import math


def get_batch(split, train_data, validation_data, batch_size, block_size, device):
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size].to(torch.long) for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size].to(torch.long) for i in ix])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, ctx, eval_iters, train_data, validation_data, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, validation_data, batch_size, block_size, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# loading pretrain gpt2 model
def load_model(out_dir, model_file, model_args=None, model_type='pretrain'):
    ckpt_path = os.path.join(out_dir, model_file)
    checkpoint = torch.load(ckpt_path, map_location='cpu')  # load from cpu, if load from gpu memo might exceed

    if model_args is not None:
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]

    else:  # just load model not rewrite args
        model_args = checkpoint['model_args']

    # create the model
    print(model_args)
    gptconf = GPTConfig(**model_args, model_type=model_type)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    if model_type == 'lora':
        model.load_weight(state_dict)
    elif model_type == 'pretrain':
        model.load_state_dict(state_dict)

    return model, checkpoint, model_args


# learning rate decay scheduler (cosine with warmup)
def get_lr_cosine_annealing(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
