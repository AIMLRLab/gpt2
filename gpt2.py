import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math
import logging

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize tokenizer and device
tokenizer = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class GPT(nn.Module):
    """
    GPT-2 Language Model Implementation (MATH.md Sections 1, 8)

    Architecture Mathematics:
    1. Input Embedding: x ∈ ℝᵈ
       - Token: E = XWᵉ
       - Position: P = pos_encode(pos)
       - Combined: H₀ = E + P

    2. Transformer Stack:
       - N layers of Ĥₗ = LN(Hₗ₋₁ + MHA(Hₗ₋₁))
       - Followed by Hₗ = LN(Ĥₗ + FFN(Ĥₗ))

    3. Output Projection:
       - Final normalization: LN(Hₙ)
       - Vocabulary projection: Wᵖ(LN(Hₙ))

    Key Parameters (MATH.md Section 8.2.2):
    - d_model: Embedding dimension (768)
    - n_heads: Number of attention heads (4)
    - n_layers: Stack depth (8)
    - block_size: Maximum sequence length (512)
    """

    def __init__(self, vocab_size=50257, d_model=768, n_heads=4, n_layers=8, block_size=512):
        super().__init__()
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing GPT-2 model on {self.device}")

        # Store model parameters
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads

        # Initialize embeddings
        logger.debug("Initializing embeddings...")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        # Initialize transformer blocks
        logger.debug(f"Creating {n_layers} transformer blocks...")
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, idx, targets=None):
        """
        Forward Pass Computation (MATH.md Section 8.1)

        Input Processing:
        1. Token embeddings (E = XWᵉ): Convert tokens to vectors
        2. Position embeddings (P): Add position information
        3. Combined (H₀ = E + P): Initial representation

        Transformer Block Processing:
        1. Multi-head attention: Ĥₗ = LN(Hₗ₋₁ + MHA(Hₗ₋₁))
        2. Feed-forward: Hₗ = LN(Ĥₗ + FFN(Ĥₗ))
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Input sequence length ({T}) exceeds block size ({self.block_size})"

        # Get embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        pos_emb = self.position_embedding(pos)

        # Add token visualization
        if logger.isEnabledFor(logging.DEBUG):
            sample_tokens = idx[0][:10].tolist()  # First sequence, first 10 tokens
            decoded = tokenizer.decode(sample_tokens)
            logger.debug("\nInput Sample:")
            logger.debug(f"Tokens: {sample_tokens}")
            logger.debug(f"Decoded: {decoded}")
            logger.debug(f"Embedding activations (mean): {tok_emb[0, :3].mean().item():.3f}")
            logger.debug(f"Position encodings (mean): {pos_emb[:3].mean().item():.3f}")

        # Combine embeddings
        x = tok_emb + pos_emb

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.output_projection(x)

        # Calculate loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Text Generation Process (MATH.md Section 3.1)

        Mathematical Process:
        1. Forward pass: Get logits from model
        2. Temperature scaling: logits/T
        3. Optional top-k: Keep only k largest probabilities
        4. Softmax: Convert to probability distribution
        5. Sample: Draw next token from distribution

        Parameters:
        - temperature: Controls randomness (default=1.0)
          - Lower (0.7): More focused predictions
          - Higher (1.2): More diverse predictions
        - top_k: Limit to k most likely tokens
          - None: Use full distribution
          - k: Only sample from top k tokens
        """
        logger.info(f"Generating {max_new_tokens} tokens with temperature={temperature}")
        if top_k:
            logger.info(f"Using top-k sampling with k={top_k}")

        for i in range(max_new_tokens):
            # Log progress every 20 tokens
            if i % 20 == 0:
                logger.debug(f"Generated {i}/{max_new_tokens} tokens")
            # Crop sequence if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def position_encoding(self, seq_len):
        """
        Sinusoidal Position Encoding (MATH.md Section 5.2)

        Mathematical Formula:
        PE(pos,2i) = sin(pos/10000^(2i/d))
        PE(pos,2i+1) = cos(pos/10000^(2i/d))

        Properties (MATH.md 5.2.2):
        1. Unique for each position
        2. Bounded between [-1, 1]
        3. Linear relationship between positions

        Example:
        Position 0: [sin(0), cos(0), sin(0/100), cos(0/100), ...]
        Position 1: [sin(1), cos(1), sin(1/100), cos(1/100), ...]
        """
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(self.device)

class TransformerBlock(nn.Module):
    """
    Transformer Block Implementation (MATH.md Section 8.1.2)

    Mathematical Flow:
    1. Multi-head attention: Ĥₗ = LN(Hₗ₋₁ + MHA(Hₗ₋₁))
    2. Feed-forward network: Hₗ = LN(Ĥₗ + FFN(Ĥₗ))
      where FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

    Each block maintains the input dimension d_model through:
    - Attention projection matrices: d_model × d_model
    - FFN expansion ratio: 4 × d_model
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        logger.debug(f"Initializing TransformerBlock with d_model={d_model}, n_heads={n_heads}")
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        # Feed-forward network (MATH.md Section 8.1.2)
        # FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        # W₁: d_model → 4*d_model
        # W₂: 4*d_model → d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class MultiHeadAttention(nn.Module):
    """
    Implementation of Multi-Head Attention (MATH.md Section 4.1)

    Mathematical Formula:
    Attention(Q,K,V) = softmax(QKᵀ/√d)V

    Where:
    - Q: Query matrix (what we're looking for)
    - K: Key matrix (what we match against)
    - V: Value matrix (what we retrieve)
    - d: Scaling factor (head_dim) to prevent dot products from growing too large

    Dimensions:
    - Input: (batch_size, seq_len, d_model)
    - Q,K,V: (batch_size, n_heads, seq_len, head_dim)
    - Output: (batch_size, seq_len, d_model)

    For detailed mathematical explanation, see:
    - MATH.md Section 4.1: Self-Attention: The Core Idea
    - MATH.md Section 4.1.1: Query, Key, Value Concept
    - MATH.md Section 4.1.2: Basic Attention Formula
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        logger.debug(f"Initializing MultiHeadAttention with d_model={d_model}, n_heads={n_heads}")
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention computation (MATH.md Section 6.1.2)
        # QKᵀ/√d_k: Scale dot products by √d_k for stable gradients
        # softmax: Convert attention scores to probabilities
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

class LayerNorm(nn.Module):
    """
    Layer Normalization Implementation (MATH.md Section 7)

    Mathematical Formula:
    x̂ = (x - μ)/√(σ² + ε)
    y = γx̂ + β

    Where:
    - μ: mean of the input
    - σ²: variance of the input
    - ε: small constant for numerical stability (1e-5)
    - γ, β: learnable parameters for scaling and shifting

    Example (MATH.md 7.1.2):
    Input [60, 70, 90]:
    1. μ = 73.33
    2. σ² = 177.78
    3. x̂ = [-1, -0.25, 1.25]
    4. y = γx̂ + β (learnable transformation)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta
