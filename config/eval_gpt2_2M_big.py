# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'eval_ckpt:out/gpt2-vanilla-295B-2M/ckpt_150000.pt'
