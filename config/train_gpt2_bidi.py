# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt2-owt'
wandb_run_name='gpt2-bidi-20B-f2'

# setup out dir
out_dir = "out/"+wandb_run_name

# 12*8 batch size * 128 block size * 1 gradaccum * 16 GPUs = 196,608
batch_size = 12*8
block_size = 128
gradient_accumulation_steps = 1 * 16

# model
bidirectional_attention = True
distribution_model = True

# this makes total number of tokens be 19,7B
warmup_iters = 2000
max_iters = 100000
lr_decay_iters = 100000
learning_rate = 5e-5 # max learning rate

# training
fmax = 2

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# dont compile
compile = False
