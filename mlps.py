import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SigmoidMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act     = nn.Sigmoid()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x) - 0.5
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class RailMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_in_vq_heads
        assert len(config.vq_block_hidden_multipliers) == 1
        
        self.rail_width: int = 16
        self.rails_per_embd = config.n_embd // self.rail_width
        self.hidden_multiplier = config.vq_block_hidden_multipliers[0]

        self.rail_maker = nn.Conv1d(
            config.n_embd,
            self.rails_per_embd * self.rail_width * self.n_heads,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.rails_per_embd
        )
        self.rail_up_proj = nn.Conv1d(
            self.rails_per_embd * self.rail_width * self.n_heads,
            self.rails_per_embd * self.rail_width * self.hidden_multiplier * self.n_heads,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.rails_per_embd * self.n_heads
        )
        self.rail_up_proj = None
       
        # self.c_proj = nn.Linear(config.n_embd * self.hidden_multiplier * self.n_heads, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd * self.n_heads, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x has shape (batch, block_size, n_embd)
        # first make the rails with rail_maker
        x = x.transpose(1, 2) # (batch, n_embd, block_size)

        x = self.rail_maker(x) # (batch, n_heads * rails_per_embd * rail_width, block_size)
        # x = self.rail_up_proj(x) # (batch, n_heads * rails_per_embd * rail_width * hidden_multiplier, block_size)
        x = F.gelu(x) # (batch, n_heads * rails_per_embd * rail_width * hidden_multiplier, block_size)
        x = x.transpose(1, 2) # (batch, block_size, n_heads * rails_per_embd * rail_width * hidden_multiplier)

        x = self.c_proj(x) # (batch, block_size, n_embd)

        x = self.dropout(x)
        return x

class DebugRailMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_in_vq_heads
        assert len(config.vq_block_hidden_multipliers) == 1
        
        self.rail_width: int = 16
        self.rails_per_embd = config.n_embd // self.rail_width
        self.hidden_multiplier = config.vq_block_hidden_multipliers[0]

        self.rail_maker = nn.Conv1d(
            config.n_embd,
            self.rails_per_embd * self.rail_width * self.n_heads,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.rails_per_embd,
            bias=config.bias,
        )
        self.rail_up_proj = nn.Conv1d(
            self.rails_per_embd * self.rail_width * self.n_heads,
            self.rails_per_embd * self.rail_width * self.hidden_multiplier * self.n_heads,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.rails_per_embd * self.n_heads,
            bias=config.bias,
        )
        self.rail_up_proj = None
       
        # self.c_proj = nn.Linear(config.n_embd * self.hidden_multiplier * self.n_heads, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd * self.n_heads, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # x has shape (batch, block_size, n_embd)
        # first make the rails with rail_maker
        x = x.transpose(1, 2) # (batch, n_embd, block_size)

        x = self.rail_maker(x) * self.rails_per_embd # (batch, n_heads * rails_per_embd * rail_width, block_size)
        x = F.gelu(x) # (batch, n_heads * rails_per_embd * rail_width * hidden_multiplier, block_size)
        x = x.transpose(1, 2) # (batch, block_size, n_heads * rails_per_embd * rail_width * hidden_multiplier)

        x = self.c_proj(x) # (batch, block_size, n_embd)

        x = self.dropout(x)
        return x

class SigmoidRailMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_in_vq_heads
        assert len(config.vq_block_hidden_multipliers) == 1
        
        self.rail_width: int = 16
        self.rails_per_embd = config.n_embd // self.rail_width
        self.hidden_multiplier = config.vq_block_hidden_multipliers[0]

        self.rail_maker = nn.Conv1d(
            config.n_embd,
            self.rails_per_embd * self.rail_width * self.n_heads,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.rails_per_embd,
            bias=config.bias,
        )
       
        self.c_proj = nn.Linear(config.n_embd * self.n_heads, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # x has shape (batch, block_size, n_embd)
        # first make the rails with rail_maker
        x = x.transpose(1, 2) # (batch, n_embd, block_size)

        x = self.rail_maker(x) * self.rails_per_embd # (batch, n_heads * rails_per_embd * rail_width, block_size)
        x = F.sigmoid(x) - 0.5 # (batch, n_heads * rails_per_embd * rail_width * hidden_multiplier, block_size)
        x = x.transpose(1, 2) # (batch, block_size, n_heads * rails_per_embd * rail_width * hidden_multiplier)

        x = self.c_proj(x) # (batch, block_size, n_embd)

        x = self.dropout(x)
        return x


class VQizer(nn.Module):
    def __init__(self, n_embd: int, n_vq_heads: int, n_vq_options: int, temperature_requires_grad: bool = True, use_temperature: bool = True):
        super().__init__()
        # n_vqheads holds the number of heads to use for vector quantization
        # n_vqoptions holds the number of vectors in the per-head codebook for vector quantization
        self.n_embd = n_embd
        self.n_vq_heads = n_vq_heads
        self.n_vq_options = n_vq_options
        self.head_size: int = n_embd // n_vq_heads

        self.vq_head_weights = nn.Parameter(torch.randn(n_vq_heads, n_vq_options, self.head_size)*0.02)
        self.vq_codebooks = nn.Parameter(torch.randn(n_vq_heads, n_vq_options, self.head_size)*0.02)

        # temperature
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=temperature_requires_grad)
        self.use_temperature = use_temperature
        self.is_frozen = False

        # tracking
        self.tracking_enabled: int = False
        self.tracking_entries: list[list[int]] = []

    def start_tracking(self) -> None:
        self.tracking_enabled = True
        self.tracking_entries = []
    
    def end_tracking(self) -> None:
        self.tracking_enabled = False

    def get_usage(self) -> list[list[int]]:
        usage = [[0 for _ in self.n_vq_options] for _ in range(self.n_vq_heads)]
        for tracking_entry in self.tracking_entries:
            for i, val in enumerate(tracking_entry):
                usage[i][val] += 1
        return usage

    def freeze(self):
        self.is_frozen = True

    def forward(self, x: torch.Tensor):
        # input shape (batch, seq_len, emb_dim)
        # matmul x and vq_head_weights to get per-head codebook choice logits
        # then apply softmax (with temperature self.t!) to get probabilities
        x_prepped = x.view(*x.shape[0:2], self.n_vq_heads, self.head_size)
        logits = torch.einsum('bsha,hoa->bsho', x_prepped, self.vq_head_weights) # shape (batch, seq_len, n_vqheads, n_vqoptions)
        
        if self.training:
            if self.use_temperature:
                probs = F.softmax(logits / self.temperature, dim=-1) # shape (batch, seq_len, n_vqheads, n_vqoptions)
            else:
                probs = F.softmax(logits, dim=-1) # shape (batch, seq_len, n_vqheads, n_vqoptions)

            if self.is_frozen:
                _, argmax = torch.max(logits, dim=-1)
                hard_probs = F.one_hot(argmax, num_classes=self.n_vq_options).to(device=x.device, dtype=logits.dtype) # shape (batch, seq_len, n_vqheads, n_vqoptions)
                probs = hard_probs + probs - probs.detach()
        else:
            # at inference time we turn the probabilities into one-hot vectors
            # this is the same as taking the argmax of the probs
            _, argmax = torch.max(logits, dim=-1)
            if self.tracking_enabled:
                flatargmax = argmax.flatten(0, -2) # shape (batch * seq_len, n_vqheads)
                # turn flatargmax into a list of n_vqoptions-lists
                self.tracking_entries.extend(flatargmax.tolist())
            probs = F.one_hot(argmax, num_classes=self.n_vq_options).to(device=x.device, dtype=logits.dtype) # shape (batch, seq_len, n_vqheads, n_vqoptions)

        # perform soft mixture by matmul of probs and codebooks
        x = torch.einsum('bsho,hoa->bsha', probs, self.vq_codebooks) # shape (batch, seq_len, n_vqheads, head_size)
        x = x.flatten(2) # shape (batch, seq_len, emb_dim)

        return x

class FastComponent(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.out_size: int = config.n_embd // config.n_vqheads
        self.depth: int = 2
        self.k: int = 4

        self.choice_linears = nn.ModuleList([nn.Linear(config.n_embd, self.k) for _ in range(self.depth)])
        self.codebook = nn.Parameter(torch.randn(self.k ** self.depth, self.out_size)*0.02)

        # temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def set_temperature(self, temperature: float) -> None:
        self.temperature.data.fill_(temperature)

    def forward(self, x: torch.Tensor):
        # input shape (batch, seq_len, emb_dim)
        choice_mixture = torch.ones(x.shape[0:2] + (self.k,) * self.depth, dtype=torch.float, device=x.device)
        for d in range(self.depth):
            choice_logits = self.choice_linears[d](x)
            choice_probs = F.softmax(choice_logits / self.temperature, dim=-1)
            choice_mixture *= choice_probs.view(*choice_probs.shape[0:2], *([1] * d + [self.k] + [1] * (self.depth - d-1)))

        choice_mixture = choice_mixture.flatten(2) # shape (batch, seq_len, k ** depth)
        out = torch.einsum('bsh,ho->bso', choice_mixture, self.codebook) # shape (batch, seq_len, out_size)

        return out

class FastMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.components = nn.ModuleList([FastComponent(config) for _ in range(config.n_vqheads)])
        self.mixer = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = torch.cat([c(x) for c in self.components], dim=-1)
        x = self.mixer(x)
        x = self.dropout(x)
        return x

class LutificationMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vq_in   = VQizer(config.n_embd, config.n_in_vq_heads, config.n_in_vq_options, config.temperature_requires_grad, config.use_temperature)
        self.vq_out  = VQizer(config.n_embd, config.n_out_vq_heads, config.n_out_vq_options, config.temperature_requires_grad, config.use_temperature)

        layers = []
        last_multiplier = 1
        for multiplier in config.vq_block_hidden_multipliers:
            layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd * multiplier, bias=config.bias))
            last_multiplier = multiplier
            layers.append(nn.GELU())
        layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd, bias=config.bias))
        self.mlp = nn.ModuleList(layers)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.vq_in(x)
        for layer in self.mlp:
            x = layer(x)
        x = self.vq_out(x)
        x = self.dropout(x)
        return x

class FSMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vq_in   = VQizer(config.n_embd, config.n_in_vq_heads, config.n_in_vq_options, config.temperature_requires_grad, config.use_temperature)

        layers = []
        last_multiplier = 1
        for multiplier in config.vq_block_hidden_multipliers:
            layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd * multiplier, bias=config.bias))
            last_multiplier = multiplier
            layers.append(nn.GELU())
        layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd, bias=config.bias))
        self.mlp = nn.ModuleList(layers)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.vq_in(x)
        for layer in self.mlp:
            x = layer(x)
        x = self.dropout(x)
        return x

class FSONMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vq_in   = VQizer(config.n_embd, config.n_in_vq_heads, config.n_in_vq_options, config.temperature_requires_grad, config.use_temperature)

        layers = []
        last_multiplier = 1
        for multiplier in config.vq_block_hidden_multipliers:
            layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd * multiplier, bias=config.bias))
            last_multiplier = multiplier
            layers.append(nn.GELU())
        layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd, bias=config.bias))
        self.mlp = nn.ModuleList(layers)
        self.on = nn.LayerNorm(config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.vq_in(x)
        for layer in self.mlp:
            x = layer(x)
        x = self.on(x)
        x = self.dropout(x)
        return x

class BitMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        last_multiplier = 1
        for multiplier in config.vq_block_hidden_multipliers:
            layers.append(nn.Linear(config.n_embd * last_multiplier, config.n_embd * multiplier, bias=config.bias))
            last_multiplier = multiplier
            layers.append(nn.GELU())
        self.mlp = nn.ModuleList(layers)
        self.c_proj = nn.Linear(config.n_embd * last_multiplier, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

        # temperature
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=config.temperature_requires_grad)
        self.use_temperature = config.use_temperature
        self.is_frozen = False

    def freeze(self):
        self.is_frozen = True

    def forward(self, x):
        if self.training:
            if self.use_temperature:
                sigms = F.sigmoid(x / self.temperature)
            else:
                sigms = F.sigmoid(x)

            if self.is_frozen:
                sigms_rounded = torch.round(sigms)
                sigms = sigms_rounded + sigms - sigms.detach()
        else:
            sigms = (x > 0).to(dtype=x.dtype)

        x = (sigms - 0.5)*0.01 # since sigms are in (0, 1), this moves them to be more balanced
        for layer in self.mlp:
            x = layer(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class PeerMLP(nn.Module):
    permutations: nn.ParameterList
    vqizers: nn.ModuleList
    mlps: nn.ModuleList

    def __init__(self, config, mlp: MLP) -> None:
        super().__init__()

        self.mlps = nn.ModuleList([mlp])
        self.vqizers = nn.ModuleList([ VQizer(config.n_embd, config.n_in_vq_heads, config.n_in_vq_options, config.temperature_requires_grad, config.use_temperature) ])
        self.permutations = nn.ParameterList([
            # add identity permutation of size config.n_embd
            nn.Parameter(torch.arange(config.n_embd, dtype=torch.long), requires_grad=False)
        ])

        # freeze all parameters in self.mlps
        for mlp in self.mlps:
            for param in mlp.parameters():
                param.requires_grad = False

    def add_peer(self, new_mlp: MLP, new_vqizer: VQizer, permutation: torch.Tensor) -> int:
        new_idx: int = len(self.mlps)
        self.mlps.append(new_mlp)
        self.vqizers.append(new_vqizer)
        self.permutations.append(nn.Parameter(permutation, requires_grad=False))

        return new_idx

    def add_new_peer(self, config) -> int:
        new_idx: int = self.add_peer(
            MLP(config),
            nn.Identity(),
            torch.randperm(config.n_embd)
        )
        return new_idx

    def add_new_linear_peer(self, config) -> int:
        new_idx: int = self.add_peer(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Identity(),
            torch.arange(config.n_embd)
        )
        self.mlps[-1].weight.data.normal_(mean=0.0, std=0.02)
        return new_idx
    
    def vqize_last(self, config) -> int:
        # freeze all parameters of mlps but the last one
        for mlp in self.mlps[:-1]:
            for param in mlp.parameters():
                param.requires_grad = False

        # freeze all parameters of all but the last vqizer
        for vqizer in self.vqizers[:-1]:
            for param in vqizer.parameters():
                param.requires_grad = False

        # replace the Identity at self.vqizers[idx] with a FullVQizer built according to config
        self.vqizers.pop(-1)
        self.vqizers.append(VQizer(config.n_embd, config.n_in_vq_heads, config.n_in_vq_options, config.temperature_requires_grad, config.use_temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, block_size, n_embd)
        y = torch.zeros_like(x)
        for permutation, vqizer, mlp in zip(self.permutations, self.vqizers, self.mlps):
            y = y + mlp(vqizer(x[:, :, permutation]))

        return y

    def unfreeze_last(self) -> None:
        # unfreeze all parameters of the last mlp
        for param in self.mlps[-1].parameters():
            param.requires_grad = True

        # unfreeze all parameters of the last vqizer unless it's "temperature"
        for name, param in self.vqizers[-1].named_parameters():
            if name != 'temperature':
                param.requires_grad = True