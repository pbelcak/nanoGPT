from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    # model specific
    model_type = "nanogpt"

    # general
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    hidden_multipliers: list[int]
    dropout: float
    bias: bool

    # vq
    vq_blocks_start: int
    vq_block_type: str
    n_in_vq_heads: int
    n_in_vq_options: int
    vq_block_hidden_multipliers: list[int]
    n_out_vq_heads: int
    n_out_vq_options: int 

    # temperature
    use_temperature: bool = True
    temperature_requires_grad: bool = False
    freezing_temperature: float = 0.0

    def __init__(self, block_size: int = 1024, vocab_size: int = 50304, n_layer: int = 12, n_head: int = 12, n_embd: int = 768,
                 hidden_multipliers: list[int] = [4], dropout: float = 0.0, bias: bool = True,
                 vq_blocks_start: int = 1000, vq_block_type: str = "fancy", n_in_vq_heads: int = 4, n_in_vq_options: int = 1024,
                 vq_block_hidden_multipliers: list[int] = [4], n_out_vq_heads: int = 4, n_out_vq_options: int = 1024,
                 use_temperature: bool = True, temperature_requires_grad: bool = False, freezing_temperature: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.hidden_multipliers = hidden_multipliers
        self.dropout = dropout
        self.bias = bias
        self.vq_blocks_start = vq_blocks_start
        self.vq_block_type = vq_block_type
        self.n_in_vq_heads = n_in_vq_heads
        self.n_in_vq_options = n_in_vq_options
        self.vq_block_hidden_multipliers = vq_block_hidden_multipliers
        self.n_out_vq_heads = n_out_vq_heads
        self.n_out_vq_options = n_out_vq_options
        self.use_temperature = use_temperature
        self.temperature_requires_grad = temperature_requires_grad
        self.freezing_temperature = freezing_temperature
