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
    GPT-2 Language Model Implementation

    Architecture Details:
    - Vocabulary Size: 50,257 tokens (GPT-2 standard)
    - Embedding Dimension: 768 (determines model capacity)
    - Number of Attention Heads: 4 (for parallel attention computation)
    - Number of Transformer Layers: 8 (depth of the model)
    - Block Size: 512 (maximum sequence length)

    The model processes text through these steps:
    1. Token Embedding: Convert tokens to vectors
    2. Position Embedding: Add position information
    3. Transformer Blocks: Process through attention layers
    4. Output Projection: Convert back to vocabulary space
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
        Forward pass of the model.

        Args:
            idx (torch.Tensor): Input token indices [batch_size, sequence_length]
            targets (torch.Tensor, optional): Target token indices for training

        Returns:
            tuple: (logits, loss) where loss is None if no targets provided
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Input sequence length ({T}) exceeds block size ({self.block_size})"

        # Get embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        pos_emb = self.position_embedding(pos)

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
        Generate text from the model.

        Args:
            idx (torch.Tensor): Starting token indices
            max_new_tokens (int): Number of tokens to generate
            temperature (float): Sampling temperature (higher = more random)
            top_k (int, optional): Limit sampling to top k tokens

        Returns:
            torch.Tensor: Generated token indices
        """
        for _ in range(max_new_tokens):
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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
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
    def __init__(self, d_model, n_heads):
        super().__init__()
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

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)
