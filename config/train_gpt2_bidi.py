# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-bidi-118B'

# setup out dir
out_dir = "out/gpt2-bidi-118B"

# 12 batch size * 1024 block size * 4 gradaccum * 16 GPUs = 786,432
batch_size = 12*8
block_size = 128
gradient_accumulation_steps = 4 * 8

# model
bidirectional_attention = True
distribution_model = True

# this makes total number of tokens be 118B
max_iters = 150000
lr_decay_iters = 150000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
