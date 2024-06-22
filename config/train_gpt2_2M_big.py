# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-vanilla-295B-2M'

# setup out dir
out_dir = "out/"+wandb_run_name

# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs * 4 nodes = 1,966,080
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8 * 4

# lr
# we had 6e-4 for ~0.5M tokens per batch, so let's have 6e-4 for ~2M tokens per batch
learning_rate = 6e-4

# weight decay
weight_decay = 1e-1

# this makes total number of tokens be 295B
max_iters = 150000
lr_decay_iters = 150000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
