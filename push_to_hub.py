"""
push_to_hub.py

A simple script to load the checkpoint specified as the first argument to the script and then push it to hugginface hub
"""

import os
import sys
import transformers
import torch

from nanogpt_model.modelling_nanogpt import GPT
from nanogpt_model.configuration_nanogpt import GPTConfig

checkpoint_path: str = sys.argv[1]

print(f"Loading a checkpoint from {checkpoint_path}")
# resume training from a checkpoint.
checkpoint = torch.load(checkpoint_path, map_location='cpu')
checkpoint_model_args = checkpoint['model_args']

# create the model
gptconf = GPTConfig(**checkpoint_model_args)
model = GPT(gptconf)

# work with the state dict
state_dict = checkpoint['model']
# - fix the keys of the state dictionary :(
# - honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
temperature = checkpoint['curr_temperature']
model.set_temperature(temperature)

# call GPT.to_pretrained to get the model in a format that can be pushed to the hub
hf_model = GPT.to_pretrained(model, 'gpt2')

# push hf_model to hub under pbelcak/nanogpt
hf_model.push_to_hub('pbelcak/nanogpt')