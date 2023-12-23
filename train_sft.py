import torch
import loralib as lora
import os
from contextlib import nullcontext
import time
from utils import get_batch, estimate_loss, load_model, get_lr_cosine_annealing


finetune_type = 'retrain_all'  # retrain_all or lora
out_dir = './out/sft'
init_from = 'sft_scratch'  # sft_scratch or sft_resume
# Create the directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)
eval_interval = 100
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'gpt2-sft'
wandb_run_name = f'gpt2 sft {time.time()}'  # 'run' + str(time.time())
# data
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 4
n_head = 4
n_embd = 384
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
lora_attn_dim = 4  # lora attn dimension
lora_attn_alpha = 32  # lora attn alpha
lora_dropout = 0.01  # dropout probability for lora layers
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 1000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False  # whether to decay the learning rate
warmup_iters = 700  # how many steps to warm up for
lr_decay_iters = 20000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for wandb logging

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)


train_data = torch.load('./data/train_data.pt')
validation_data = torch.load('./data/validation_data.pt')


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout,
                  lora_attn_dim=lora_attn_dim, lora_attn_alpha=lora_attn_alpha, lora_dropout=lora_dropout)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

if init_from == 'sft_scratch':
    print(f"SFT from scratch with finetune type {finetune_type}")
    if finetune_type == 'retrain_all':
        model, _, model_args = load_model(out_dir, 'ckpt_pretrain.pt', model_args, 'pretrain')
    elif finetune_type == 'lora':
        model, _, model_args = load_model(out_dir, 'ckpt_pretrain.pt', model_args, 'lora')
elif init_from == 'sft_resume':
    print(f"Resuming training from {out_dir} with finetune type {finetune_type}")
    if finetune_type == 'retrain_all': # better named retrain_all, could try tranfer learning
        model, checkpoint, model_args = load_model(out_dir, 'ckpt_sft.pt', model_args, 'pretrain')
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif finetune_type == 'lora':
        model, _, model_args = load_model(out_dir, 'ckpt_pretrain.pt', model_args, 'lora')
        lora_ckpt_path = os.path.join(out_dir, 'ckpt_lora.pt')
        checkpoint = torch.load(lora_ckpt_path, map_location='cpu')
        lora_w = checkpoint['model']
        model.load_state_dict(lora_w, strict=False)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

model.to(device)
if lora_attn_dim > 0 and finetune_type == 'lora':
    lora.mark_only_lora_as_trainable(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)


# logging
if wandb_log:
    import wandb

    wandb.login()  # my wandb account key, change to your own
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train', train_data, validation_data, batch_size, block_size, device)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr_cosine_annealing(iter_num, learning_rate, min_lr, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model, ctx, eval_iters, train_data, validation_data, batch_size, block_size, device)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict() if finetune_type == 'retrain_all' else lora.lora_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_sft.pt'
                                                    if finetune_type == 'retrain_all' else 'ckpt_lora.pt'))


    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', train_data, validation_data, batch_size, block_size, device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt:.2f}s, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
