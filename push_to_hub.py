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
huggingface_name: str = sys.argv[2]

print(f"Loading a checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
checkpoint_model_args = checkpoint['model_args']

def fix_state_dict(state_dict):
    # - fix the keys of the state dictionary :(
    # - honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

def load_nano_as_hf():
    # create the model
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)

    # work with the state dict
    state_dict = checkpoint['model']
    fix_state_dict(state_dict)
    model.load_state_dict(state_dict)
    temperature = checkpoint['curr_temperature']
    model.set_temperature(temperature)

    # call GPT.to_pretrained to get the model in a format that can be pushed to the hub
    hf_model = GPT.to_pretrained(model, 'gpt2')
    return hf_model

def load_hf():
    hf_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    state_dict = checkpoint['model']
    fix_state_dict(state_dict)
    hf_model.load_state_dict(state_dict)
    return hf_model

hf_model = load_hf()

# push hf_model to hub under pbelcak/nanogpt
print(f"Pushing this checkpoint to HF hub as {huggingface_name}")
hf_model.push_to_hub(f"pbelcak/{huggingface_name}")

# load pre-trained GPT2 tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

# push it to hub under pbelcak/nanogpt
tokenizer.push_to_hub(f"pbelcak/{huggingface_name}")
