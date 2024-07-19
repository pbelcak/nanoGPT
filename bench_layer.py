"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import time
import numpy as np
import torch
from nanogpt_model.modelling_nanogpt import GPT
from nanogpt_model.configuration_nanogpt import GPTConfig

from torch import nn

# -----------------------------------------------------------------------------
# surgeries
surgeries = None # no surgeries by default
past_surgeries = None # no past surgeries by default
# data
dataset = 'openwebtext'
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
hidden_multipliers: list[int] = [4]
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# model vq setup
vq_blocks_start = 1000
vq_block_type: str = "fancy"
n_in_vq_heads = 4
n_in_vq_options = 1024
vq_block_hidden_multipliers: list[int] = [4]
n_out_vq_heads = 4
n_out_vq_options = 1024
# temperature setup
use_temperature = False
temperature_requires_grad = False
start_temperature = 1.0
end_temperature = 0.01
freezing_temperature = 0.00

real_data = True
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
import surgery

# data loading init
if real_data:
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    def get_batch(split):
        return x, y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_hidden = config.n_embd * 4
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SmallMLP(nn.Module):
    def __init__(self, config, n_hidden):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_hidden = n_hidden
        self.fc    = nn.Linear(config.n_embd, n_hidden, bias=config.bias)
        self.gelu    = nn.GELU()
        self.proj  = nn.Linear(n_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class VQizedMLP(nn.Module):
    def __init__(self, n_embd: int, n_hidden: int, n_vq_heads: int, n_vq_options: int):
        super().__init__()
        # n_vqheads holds the number of heads to use for vector quantization
        # n_vqoptions holds the number of vectors in the per-head codebook for vector quantization
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.n_vq_heads = n_vq_heads
        self.n_vq_options = n_vq_options
        self.head_size: int = n_embd // n_vq_heads

        self.vq_head_weights = nn.Parameter(torch.randn(n_vq_heads, n_vq_options, self.n_embd)*0.02)
        self.vq_codebooks = nn.Parameter(torch.randn(n_vq_heads, n_vq_options, self.n_embd)*0.02)
        
        self.table_size = n_vq_options ** n_vq_heads
        self.linear1_table = nn.Parameter(torch.randn(self.table_size, n_embd * n_hidden, dtype=torch.bfloat16)*0.02)
        self.linear2_table = nn.Parameter(torch.randn(self.table_size, n_hidden * n_embd, dtype=torch.bfloat16)*0.02)

    def forward(self, x: torch.Tensor):
        logits = torch.einsum('bse,hoe->bsho', x, self.vq_head_weights) # shape (batch, seq_len, n_vqheads, n_vqoptions)
    
        head_indices = torch.max(logits, dim=-1).indices

        x = x.flatten(0, 1).unsqueeze(1) # shape (batch * block_size, 1, n_embd)
        head_indices_flat = head_indices.flatten(0, 1) # shape (batch * block_size, n_heads)
        
        # convert head_indices_flat into absolute indices
        flat_indices = torch.zeros((x.shape[0] * x.shape[1],), dtype=torch.long, device=x.device) # shape (batch, block_size, n_heads)
        multiplier: int = 1
        for i in range(self.n_vq_heads):
            flat_indices += multiplier * head_indices_flat[:, i]
            multiplier *= self.n_vq_options
    
        # index select the table
        linear1 = torch.index_select(self.linear1_table, 0, flat_indices) # shape (batch * block_size, n_embd * n_hidden)
        linear1 = linear1.view(-1, self.n_embd, self.n_hidden)
        y = torch.bmm(x, linear1) # shape (batch * block_size, 1, n_hidden)

        y = torch.nn.functional.gelu(y)

        linear2 = torch.index_select(self.linear2_table, 0, flat_indices)
        linear2 = linear2.view(-1, self.n_hidden, self.n_embd)
        y = torch.bmm(y, linear2) # shape (batch * block_size, 1, n_embd)

        y = y.view(*head_indices.shape[0:2], self.n_embd)

        return y

# model init
gptconf = GPTConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    hidden_multipliers=hidden_multipliers,
    block_size=block_size,
    bias=bias,
    dropout=dropout,

    vq_blocks_start=vq_blocks_start,
    vq_block_type=vq_block_type,
    n_in_vq_heads=n_in_vq_heads,
    n_in_vq_options=n_in_vq_options,
    vq_block_hidden_multipliers=vq_block_hidden_multipliers,
    n_out_vq_heads=n_out_vq_heads,
    n_out_vq_options=n_out_vq_options,

    use_temperature=use_temperature,
    temperature_requires_grad=temperature_requires_grad,
    freezing_temperature=freezing_temperature,
)
#model = GPT(gptconf)
#if past_surgeries is not None and len(past_surgeries) > 0:
#    surgery.perform_surgeries(gptconf, model, past_surgeries)
#if surgeries is not None and len(surgeries) > 0:
#    surgery.perform_surgeries(gptconf, model, surgeries)
#model = MLP(gptconf)
#model = SmallMLP(gptconf, 512)
model = VQizedMLP(768, 512, 4, 8)
model.to(device)

compile = True
if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

X = torch.randn(batch_size, block_size, 768, device=device)
if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        for k in range(num_steps):
            with ctx:
                model(X)

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 100]): # burnin, then benchmark
        t0 = time.time()
        for k in range(num_steps):
            with ctx:
                model(X)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms")
