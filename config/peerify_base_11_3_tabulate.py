# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt2-peerify'
wandb_run_name='peerify_base_11_3_32_tabulate'

# setup out dir
out_dir = "out/"+wandb_run_name
standalone_ckpt_frequency = 2500

# init
init_from = 'peerify_ckpt:out/peerify_base_11_1_32/ckpt_10000.pt'
past_surgeries = [
    ('peerify', 11),
]
surgeries = [
    ('tabulate_last', 11),
]

# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs * 4 nodes = ~2M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8 * 4

# lr
# we had 6e-4 for ~0.5M tokens per batch
learning_rate = 6e-4
table_learning_rate = 6e-3

# weight decay
weight_decay = 1e-1

# this makes total number of tokens be ~20B
warmup_iters = 1000
max_iters = 10000
lr_decay_iters = 10000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# temperature
use_temperature = False
temperature_requires_grad = False
start_temperature = 1.0
end_temperature = 0.05
freezing_temperature = 1.0

# vq config
n_in_vq_heads = 4
n_in_vq_options = 32

# compilation switch for testing
compile = False