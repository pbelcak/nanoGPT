import torch
import inspect

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, table_learning_rate):
    # start with all of the candidate parameters
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad and not pn.endswith('.table')}
    table_param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad and pn.endswith('.table')}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    num_table_params = sum(p.numel() for pn, p in table_param_dict.items())
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"num table tensors {len(table_param_dict)}, with {num_table_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    table_optimizer = torch.optim.AdamW(table_param_dict.values(), lr=table_learning_rate, betas=betas, **extra_args) if len(table_param_dict) > 0 else None
    print(f"using fused AdamW: {use_fused}")

    return optimizer, table_optimizer