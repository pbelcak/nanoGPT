# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'

# setup out dir
out_dir = "out/gpt2-lut-118B-11-4x1024"
