wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-debugrailmlp-0-4-1'


# setup out dir
out_dir = "out/"+wandb_run_name

# 12 batch size * 1024 block size * 4 gradaccum * 16 GPUs = 393,216
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 2 * 16

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
vq_block_type = "debug-rail-mlp"
n_in_vq_heads = 4
vq_block_hidden_multipliers: list[int] = [1]

# temperature
use_temperature = True
temperature_requires_grad = False
start_temperature = 1.0
end_temperature = 0.05
freezing_temperature = 0.90

# this makes total number of tokens be 118B
max_iters = 150000
lr_decay_iters = 150000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# compilation
compile = False