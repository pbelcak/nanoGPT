# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'eval_ckpt'

# setup out dir
out_dir = "out/gpt2-lut-118B-11-4x1024"

# model
n_layer = 12
n_head = 12
n_embd = 768
n_hidden_multiplier = 6
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vq_blocks_start = 11
n_vqheads = 4
n_vqoptions = 1024
