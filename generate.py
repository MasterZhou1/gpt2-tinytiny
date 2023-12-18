"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
from transformers import GPT2Tokenizer
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
model_type = 'pretrain' # the type of model used for generate new sentences. 'pretrain' or 'lora' or 'sft'
start = "The white house is located in" # Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 64 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# model loading
def load_model(out_dir, model_file):
    ckpt_path = os.path.join(out_dir, model_file)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'], model_type='pretrain')
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict, model

# init from a model saved in a specific directory
if model_type == 'pretrain':
    out_dir = 'out'
    state_dict, model = load_model(out_dir, 'ckpt.pt')
    model.load_state_dict(state_dict)
elif model_type == 'sft':
    out_dir = './out/sft'
    state_dict, model = load_model(out_dir, 'ckpt_sft.pt')
    model.load_weight(state_dict)
elif model_type == 'lora':
    out_dir = './out/sft'
    state_dict, model = load_model(out_dir, 'ckpt_pretrain.pt')
    model.load_weight(state_dict)
    lora_ckpt_path = os.path.join(out_dir, 'ckpt_sft_lora.pt')
    lora_w = torch.load(lora_ckpt_path, map_location=device)
    model.load_state_dict(lora_w, strict=False)


model.eval()
model.to(device)


# load GPT2 encoder
enc = GPT2Tokenizer.from_pretrained("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')