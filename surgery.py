from nanogpt_model.modelling_nanogpt import Block, MLP
from nanogpt_model.mlps import PeerMLP, SmallMLP

import torch

def perform_surgeries(config, model, surgeries):
    for surgery_type, layer in surgeries:
        if surgery_type == 'freeze_all':
            freeze_all(config, model)
        elif surgery_type == 'unfreeze_all':
            unfreeze_all(config, model)
        elif surgery_type == 'freeze_mlp':
            freeze_mlp(config, model, layer)
        elif surgery_type == 'peerify':
            peerify(config, model, layer)
        elif surgery_type == "add_peer_mlp":
            add_peer_mlp(config, model, layer)
        elif surgery_type == "add_peer_linear":
            add_peer_linear(config, model, layer)
        elif surgery_type == 'vqize_last':
            vqize_last(config, model, layer)
        elif surgery_type == 'fullvqize_last':
            fullvqize_last(config, model, layer)
        elif surgery_type == 'tabulate_last':
            tabulate_last(config, model, layer)
        elif surgery_type == 'unfreeze_last':
            unfreeze_last(config, model, layer)
        elif surgery_type == 'pte_last':
            pte_last(config, model, layer)
        elif surgery_type == 'smallmlp512':
            smallmlp512(config, model, layer)
        else:
            raise ValueError(f"Unknown surgery type: {surgery_type}")

def freeze_all(config, model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    print("Frozen all the model parameters")
    
def unfreeze_all(config, model) -> None:
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            param.requires_grad = True
    print("Unfrozen all the model parameters")

def freeze_mlp(config, model, layer: int) -> None:
    tgt_block: Block = model.transformer.h[layer]
    for param in tgt_block.mlp.parameters():
        param.requires_grad = False
    print("Frozen the mlp of block ", layer)

def smallmlp512(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]

    tgt_block.mlp = SmallMLP(config, 512)
    print("Changed the last mlp of the block ", block_idx, " to have hidden size 512")

def peerify(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    # print(tgt_block)
    if isinstance(tgt_block.mlp, MLP):
        tgt_block.mlp = PeerMLP(config, tgt_block.mlp)
    else:
        raise ValueError(f"Block {block_idx} is not a peerifiable block")
    
    print("Peerified block ", block_idx)

def add_peer_mlp(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")

    tgt_block.mlp.add_new_peer(config)
    # go through all parameters of all vqizers and freeze them
    for vqizer in tgt_block.mlp.vqizers:
        for param in vqizer.parameters():
            param.requires_grad = False

    # freeze all parameters of all but the last mlp
    for mlp in tgt_block.mlp.mlps[:-1]:
        for param in mlp.parameters():
            param.requires_grad = False
    
    print("Added a peer to the PeerMLP of block ", block_idx)

def add_peer_linear(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")

    tgt_block.mlp.add_new_linear_peer(config)
    # go through all parameters of all vqizers and freeze them
    for vqizer in tgt_block.mlp.vqizers:
        for param in vqizer.parameters():
            param.requires_grad = False

    # freeze all parameters of all but the last mlp
    for mlp in tgt_block.mlp.mlps[:-1]:
        for param in mlp.parameters():
            param.requires_grad = False
    
    print("Added a peer linear to the PeerMLP of block ", block_idx)

def vqize_last(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")
    
    tgt_block.mlp.vqize_last(config)
    print("VQized the last mlp of the PeerMLP of block  ", block_idx)

def fullvqize_last(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")
    
    tgt_block.mlp.fullvqize_last(config)
    print("VQized the last mlp of the PeerMLP of block  ", block_idx)

def unfreeze_last(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")
    
    tgt_block.mlp.unfreeze_last()
    print("Flip-froze the last mlp of the PeerMLP of block  ", block_idx)

def tabulate_last(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp, it has {tgt_block.mlp}")

    tgt_block.mlp.tabulate_last()
    print("Tabulated the last mlp of the PeerMLP of block  ", block_idx)

def pte_last(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")
    
    tgt_block.mlp.pte_last()
    print("PTEd the last mlp of the PeerMLP of block  ", block_idx)