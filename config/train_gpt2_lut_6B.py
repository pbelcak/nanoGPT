# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-lut-118B-6-8x1024-8x1024'

# setup out dir
out_dir = "out/"+wandb_run_name

# 12 batch size * 1024 block size * 4 gradaccum * 16 GPUs = 786,432
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 4 * 16

# model
n_layer = 12
n_head = 12
n_embd = 768
hidden_multipliers: list[int] = [4]
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
vq_blocks_start = 6
n_in_vq_heads = 8
n_in_vq_options = 1024
vq_block_hidden_multipliers: list[int] = [4]
n_out_vq_heads = 8
n_out_vq_options = 1024

# this makes total number of tokens be 118B
max_iters = 150000
lr_decay_iters = 150000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
