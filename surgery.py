from model import *
from mlps import *

def perform_surgeries(config, model, surgeries):
    for surgery_type, layer in surgeries:
        if surgery_type == 'peerify':
            peerify(config, model, layer)
        elif surgery_type == "add_peer_mlp":
            add_peer_mlp(config, model, layer)
        elif surgery_type == "add_peer_linear":
            add_peer_linear(config, model, layer)
        elif surgery_type == 'vqize_last':
            vqize(config, model, layer)
        elif surgery_type == 'tabulate_last':
            tabulate_last(config, model, layer)
        elif surgery_type == 'unfreeze_last':
            unfreeze_last(config, model, layer)
        else:
            raise ValueError(f"Unknown surgery type: {surgery_type}")


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

def vqize(config, model, block_idx: int) -> None:
    tgt_block: Block = model.transformer.h[block_idx]
    if not isinstance(tgt_block.mlp, PeerMLP):
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")
    
    tgt_block.mlp.vqize_last(config)
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
        raise ValueError(f"Block {block_idx} does not have a PeerMLP as mlp")

    tgt_block.mlp.tabulate_last()
    print("Tabulated the last mlp of the PeerMLP of block  ", block_idx)