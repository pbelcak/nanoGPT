wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-fs-0-4x256-4-big'

# setup out dir
out_dir = "out/"+wandb_run_name

# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs * 4 nodes = 1,966,080
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8 * 4

# lr
# we had 6e-4 for ~0.5M tokens per batch
learning_rate = 6e-4

# weight decay
weight_decay = 1e-1

# model
n_layer = 12
n_head = 12
n_embd = 768
hidden_multipliers: list[int] = [4]
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
vq_blocks_start = 0
vq_block_type = "fs-mlp"
n_in_vq_heads = 4
n_in_vq_options = 256
vq_block_hidden_multipliers: list[int] = [4]

# temperature
use_temperature = False
temperature_requires_grad = False
start_temperature = 1.0
end_temperature = 0.05
freezing_temperature = 0.90

# this makes total number of tokens be 295B
max_iters = 150000
lr_decay_iters = 150000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# compilation
compile = False