# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 500
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-2'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
vq_blocks_start = 3
hidden_multipliers: list[int] = [4]
bias = False
vq_block_type = "fs-mlp"
n_in_vq_heads = 3
n_in_vq_options = 256
vq_block_hidden_multipliers: list[int] = [4,4]

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# temperature setup
start_temperature = 1.0
end_temperature = 0.01
freezing_temperature = 0.80

# on macbook also add
# device = 'cpu'  # run on cpu only
quick_debug = True
compile = False # do not torch compile the model
