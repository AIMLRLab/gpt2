# ===================================
# GPT-2 Implementation for Learning
# ===================================
# This implementation is a simplified version of the GPT-2 architecture
# designed for educational purposes. Each component is extensively documented
# to explain the underlying concepts and implementation details.

# =================
# 1. IMPORTS
# =================
import time                      # For tracking training time
import math                      # For mathematical operations (e.g., sqrt in attention)
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch                     # Main PyTorch library
import tiktoken                  # OpenAI's tokenizer for GPT-2
import matplotlib.pyplot as plt  # For visualizing training progress
import numpy as np
from typing import List, Tuple, Dict, Optional
import seaborn as sns  # For better visualizations
from torch.utils.data import DataLoader as TorchDataLoader  # Rename to avoid conflict

# =================
# 2. INITIAL SETUP
# =================
# Initialize the tokenizer with GPT-2's vocabulary
tokenizer = tiktoken.get_encoding("gpt2")
vocabulary_size = tokenizer.n_vocab  # Number of tokens in GPT-2's vocabulary

# Set the device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =================
# 3. CONFIGURATION
# =================
# Training hyperparameters
training_batch_size = 16    # Number of sequences processed in parallel during training
evaluation_batch_size = 8   # Number of sequences processed in parallel during evaluation
sequence_length = 512       # Maximum number of tokens in a sequence
training_split = 0.7       # Percentage of data used for training (70% train, 30% validation)

# Model architecture parameters
embedding_dimension = 768   # Size of token embeddings and model's hidden layers
attention_heads = 4        # Number of attention heads in multi-head attention
transformer_layers = 8     # Number of transformer blocks in the model

# Training parameters
total_epochs = 5        # Total number of training epochs
evaluation_interval = 500  # How often to evaluate the model
learning_rate = 1e-3      # Initial learning rate for optimization

# =================
# 4. DATA LOADING
# =================
# Load and preprocess the training data
data_filepath = "data.txt"
print(f"\nLoading training data from: {data_filepath}")

# Read the entire text file
raw_text = open(data_filepath, "r").read()
print(f"Total characters in dataset: {len(raw_text):,}")

# Convert text to token IDs using GPT-2 tokenizer
token_ids = torch.tensor(
    tokenizer.encode(raw_text),  # Encode text to token IDs
    dtype=torch.long,            # Use long integer type for token IDs
    device=device               # Place tensor on appropriate device
)
print(f"Total tokens after encoding: {len(token_ids):,}")

# =================
# 5. DATA SPLITTING
# =================
# Split data into training and validation sets
total_tokens = len(token_ids)
split_index = int(total_tokens * training_split)

# Create training and validation datasets
training_data = token_ids[:split_index]
validation_data = token_ids[split_index:]

print(f"\nData Split:")
print(f"Training tokens: {len(training_data):,}")
print(f"Validation tokens: {len(validation_data):,}")

# 6. Model Components (in order of dependency)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        print(f"\n{'='*80}")
        print("🎯 Understanding Multi-Head Attention")
        print(f"{'='*80}")
        print("\n🤔 What is Attention?")
        print("Imagine you're reading a book and some words are highlighted with different colors.")
        print("Each color (head) helps you focus on different parts of the text!")
        print("\nExample:")
        print("'The cat sat on the mat'")
        print("- Head 1 might focus on: who? (the cat)")
        print("- Head 2 might focus on: what did they do? (sat)")
        print("- Head 3 might focus on: where? (on the mat)")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        print("\n📊 Let's Break Down the Numbers:")
        print(f"• Number of attention heads: {n_heads}")
        print(f"   └── Like having {n_heads} different ways to look at the text")
        print(f"• Each head dimension: {self.head_dim}")
        print(f"   └── Each way of looking uses {self.head_dim} numbers to understand the text")
        print(f"• Total dimension: {d_model}")
        print(f"   └── All together, we use {d_model} numbers to represent each word")

        # Create attention components
        print("\n🔨 Building the Attention Tools:")
        print("1. Query: Like asking questions about the text")
        print("2. Key: Like an index to find relevant information")
        print("3. Value: The actual information we want to gather")

        print("\n💡 Real-World Example:")
        print("When you Google search:")
        print("• Query = Your search terms")
        print("• Keys = Website titles and descriptions")
        print("• Values = The actual websites you get back")

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        # Initialize dropout BEFORE using it
        print("\n🎲 Adding Dropout Layer:")
        print("   └── Like randomly covering words while reading")
        print("   └── Helps prevent memorization (20% chance to ignore connections)")
        self.dropout = nn.Dropout(0.2)  # Make sure this is initialized!

        # Store attention weights
        self.store_attention_weights = False
        self.last_attention_weights = None

    def forward(self, inputs: torch.Tensor):
        """Process input through attention mechanism."""
        try:
            B, seq_length, d_model = inputs.shape

            # Project inputs to Q, K, V
            Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # Compute attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Create and apply causal mask
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))

            # Compute attention weights with dropout
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Store attention weights if requested
            if self.store_attention_weights:
                self.last_attention_weights = attention_weights.detach()

            # Compute weighted sum of values
            attention_output = torch.matmul(attention_weights, V)

            # Reshape and project output
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(B, seq_length, d_model)
            return self.fc_out(attention_output)

        except Exception as e:
            print(f"\n❌ Error in attention mechanism:")
            print(f"   └── {str(e)}")
            print(f"   └── Input shape: {inputs.shape}")
            raise  # Re-raise the exception after logging

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model):
        super().__init__()
        print(f"\n{'='*80}")
        print("🔢 Understanding Position Encoding")
        print(f"{'='*80}")

        print("\n🤔 Why Do We Need This?")
        print("Imagine reading a sentence with all words jumbled:")
        print("'cat the mat on sat'")
        print("It's hard to understand, right? Order matters!")
        print("\nThe model needs to know word positions, just like you do.")

        print("\n📏 How It Works:")
        print(f"• Context Length: {context_length} tokens")
        print(f"   └── Can remember up to {context_length} words at once")
        print(f"• Each position uses {d_model} numbers")
        print("• Uses special math (sine/cosine waves) to create unique position codes")

        print("\n🎨 Visual Example:")
        print("Position 1: 📍")
        print("Position 2: 📍📍")
        print("Position 3: 📍📍📍")
        print("Each position gets a unique pattern, like a special stamp!")

        # Create position encoding matrix
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        print("\n📝 EXPLANATION: Positional Encoding Steps")
        print("1. Create a matrix of zeros: (context_length × d_model)")
        print("2. Fill even indices with sine waves")
        print("3. Fill odd indices with cosine waves")
        print("4. Each position gets unique timing signals")

        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # add batch dimension

        # Register buffer (won't be updated during training)
        self.register_buffer('pe', pe)

        # Show example of first few positions
        print("\n🔍 Example of first 3 positions (first 8 dimensions):")
        for pos in range(min(3, context_length)):
            print(f"Position {pos}: {pe[0, pos, :8].tolist()}")

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        print(f"\n{'='*80}")
        print("🧱 Understanding GPT Blocks - The Building Blocks of AI")
        print(f"{'='*80}")

        print("\n🏗️ What is a GPT Block?")
        print("Think of it like a LEGO piece in our AI building:")
        print("1. First it looks at words (Attention)")
        print("2. Then it thinks about what it saw (Neural Network)")
        print("3. Finally it organizes its thoughts (Normalization)")

        print("\n🔍 Block Components:")
        self.att = MultiHeadAttention(d_model, n_heads)
        print("\n1️⃣ First Layer Normalization")
        print("   └── Like organizing thoughts before processing")
        self.ln1 = nn.LayerNorm(d_model)

        print("\n2️⃣ Second Layer Normalization")
        print("   └── Like organizing thoughts after processing")
        self.ln2 = nn.LayerNorm(d_model)

        print("\n3️⃣ Feed Forward Network")
        print("   └── The 'thinking' part of the AI")
        print(f"   └── Input size: {d_model}")
        print(f"   └── Hidden size: {4 * d_model}")
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        print("\n4️⃣ Dropout - Preventing Memorization")
        print("   └── Like covering parts of a textbook to test understanding")
        print("   └── Helps the model learn patterns, not memorize")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Process input through the transformer block."""
        try:
            # Attention with residual connection
            att_out = self.att(x)
            x1 = self.ln1(x + att_out)

            # Feed-forward with residual connection
            ff_out = self.fcn(x1)
            x2 = self.ln2(ff_out + x1)

            # Apply dropout
            return self.dropout(x2)

        except Exception as e:
            print(f"\n❌ Error in GPT Block:")
            print(f"   └── {str(e)}")
            print(f"   └── Input shape: {x.shape}")
            raise

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        print(f"\n{'='*80}")
        print(f"🏗️ GPT Architecture Visualization")
        print(f"{'='*80}")

        print("\n1️⃣ Input Layer")
        print(f"   └── Vocabulary Size: {vocab_size}")
        print(f"   └── Embedding Dimension: {d_model}")
        self.wte = nn.Embedding(vocab_size, d_model)

        print("\n2️⃣ Positional Encoding")
        print(f"   └── Context Length: {sequence_length}")
        print(f"   └── Embedding Dimension: {d_model}")
        self.wpe = PositionalEncoding(sequence_length, d_model)

        print("\n3️⃣ Transformer Blocks")
        print(f"   └── Number of Layers: {n_layers}")
        print(f"   └── Number of Heads: {n_heads}")
        print(f"   └── Head Dimension: {d_model // n_heads}")
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])

        print("\n4️⃣ Output Layer")
        print(f"   └── Final Linear: {d_model} → {vocab_size}")
        self.linear1 = nn.Linear(d_model, vocab_size)

        # Parameter sharing
        self.wte.weight = self.linear1.weight

        # Store architecture parameters for attention visualization
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model

        # Calculate model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Model Summary:")
        print(f"   └── Total Parameters: {total_params:,}")
        print(f"   └── Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

    def get_attention_weights(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Captures attention weights from all layers during a forward pass.

        Args:
            input_ids: Input token IDs [batch_size, sequence_length]

        Returns:
            List of attention weight tensors for each layer
            Each tensor shape: [batch_size, n_heads, sequence_length, sequence_length]
        """
        attention_weights = []

        # Initial embeddings
        x = self.wte(input_ids)
        x = self.wpe(x)

        # Pass through each transformer block
        for block in self.blocks:
            # Store the attention weights for this layer
            block.att.store_attention_weights = True
            x = block(x)
            attention_weights.append(block.att.last_attention_weights)
            block.att.store_attention_weights = False

        return attention_weights

    def forward(self, inputs, targets=None):
        try:
            # Token embeddings
            logits = self.wte(inputs)
            logits = self.wpe(logits)

            # Pass through transformer blocks
            for block in self.blocks:
                logits = block(logits)

            # Final linear layer
            logits = self.linear1(logits)

            # Calculate loss if targets provided
            loss = None
            if targets is not None:
                batch_size, sequence_length, d_model = logits.shape
                logits = logits.view(batch_size * sequence_length, d_model)
                targets = targets.view(batch_size * sequence_length)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        except Exception as e:
            print(f"\n❌ Error in forward pass:")
            print(f"   └── {str(e)}")
            print(f"   └── Input shape: {inputs.shape}")
            if targets is not None:
                print(f"   └── Target shape: {targets.shape}")
            raise  # Re-raise the exception after logging

class CustomDataLoader:
    """
    Custom data loader for handling token sequences.
    Provides batches of data for training and evaluation.
    """
    def __init__(self, tokens, batch_size, context_length, verbose=False) -> None:
        self.verbose = verbose
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔍 UNDERSTANDING THE DATALOADER INITIALIZATION")
            print(f"{'='*80}")

            print(f"\n📚 STEP 1: Understanding Our Input Text Data")
            print(f"Our raw text has been converted to {len(tokens):,} tokens")
            sample_text = ''.join([tokenizer.decode([t.item()]) for t in tokens[:50]])
            print(f"First 50 characters of text: '{sample_text}'")
            print(f"Same text as tokens: {tokens[:50].tolist()}")
            print("\n📝 EXPLANATION: Each character in our text has been converted to a number (token).")
            print("This makes it easier for the model to process the text.")

            print(f"\n🧮 STEP 2: Understanding Batch Processing")
            print(f"Batch size: {batch_size} (We process {batch_size} sequences at once)")
            print(f"Context length: {context_length} (Each sequence is {context_length} tokens long)")
            print(f"Therefore, each batch contains: {batch_size} × {context_length} = {batch_size * context_length:,} tokens")
            print("\n📝 EXPLANATION: Imagine dividing a book into:")
            print(f"- {batch_size} parallel reading streams")
            print(f"- Each stream contains {context_length} characters")
            print(f"- The model learns to predict the next character in each stream")

            print(f"\n💾 STEP 3: Memory and Processing Details")
            bytes_per_batch = batch_size * context_length * tokens.element_size()
            print(f"Memory per batch: {bytes_per_batch/1024:.1f} KB")
            total_batches = len(tokens) // (batch_size * context_length)
            print(f"Total complete batches possible: {total_batches:,}")

        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔄 CREATING A NEW BATCH OF TRAINING DATA")
            print(f"{'='*80}")

            print(f"\n📍 STEP 1: Position Check")
            print(f"Currently at position: {self.current_position:,} of {len(self.tokens):,} tokens")
            progress = (self.current_position / len(self.tokens)) * 100
            print(f"Progress through text: {progress:.1f}%")
            print("█" * int(progress/5) + "░" * (20 - int(progress/5)))

        batch, context = self.batch_size, self.context_length
        start_pos = self.current_position
        total_tokens_needed = batch * context
        end_pos = start_pos + total_tokens_needed

        tokens_remaining = len(self.tokens) - start_pos
        if tokens_remaining < total_tokens_needed + 1:
            self.current_position = 0
            start_pos = 0
            end_pos = total_tokens_needed

        d = self.tokens[start_pos:end_pos + 1]

        if self.verbose:
            print(f"\n🎯 STEP 3: Creating Training Pairs")
            print("For each sequence, we create:")
            print("1. Input:  tokens[0   to 255]  → Model sees this")
            print("2. Target: tokens[1   to 256]  → Model should predict this")
            print("\nExample from first sequence:")
            input_text = ''.join([tokenizer.decode([t.item()]) for t in d[:10]])
            target_text = ''.join([tokenizer.decode([t.item()]) for t in d[1:11]])
            print(f"Input:  '{input_text}'")
            print(f"Target: '{target_text}'")
            print("\n📝 EXPLANATION: The model learns to predict the next character at each position")

        x = d[:-1].view(batch, context)
        y = d[1:].view(batch, context)

        if self.verbose:
            print(f"\n✅ STEP 4: Final Batch Shape")
            print(f"Input shape:  {tuple(x.shape)} (batch_size, context_length)")
            print(f"Target shape: {tuple(y.shape)} (batch_size, context_length)")

        self.current_position = end_pos
        return x, y

class ModelEvaluator:
    """
    Handles model evaluation, metrics tracking, and visualization during training.
    Provides detailed insights into model behavior and performance.
    """
    def __init__(self, model: nn.Module, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Metrics tracking
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.perplexity_history = []
        self.token_accuracy = []

        # Visualization settings
        try:
            plt.style.use('seaborn-v0_8')  # Use the updated seaborn style name
        except Exception as e:
            print(f"Warning: Could not set seaborn style: {e}")
            print("Falling back to default matplotlib style")

        self.figure_size = (12, 8)

    def evaluate_loss(self, data_loader: CustomDataLoader) -> float:
        """Calculate average loss with progress tracking."""
        print("\n📊 Evaluating Model Performance")
        total_loss = 0.0
        num_batches = 0
        best_loss = float('inf')

        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(100):
                input_ids, target_ids = data_loader.get_batch()
                logits, loss = self.model(input_ids, target_ids)
                total_loss += loss.item()
                num_batches += 1

                current_avg = total_loss / num_batches
                if current_avg < best_loss:
                    best_loss = current_avg

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/100: Loss = {loss.item():.4f} (Best: {best_loss:.4f})")

        average_loss = total_loss / num_batches
        print(f"\nFinal Average Loss: {average_loss:.4f}")
        return average_loss

    def visualize_attention(self, input_text: str, layer_idx: int = 0, head_idx: int = 0, ax=None):
        """
        Visualize attention patterns for a given input.
        Shows which tokens the model is focusing on.
        """
        print(f"\n🔍 Visualizing Attention Patterns")
        print(f"Input text: '{input_text}'")

        try:
            # Tokenize input
            input_ids = torch.tensor(
                self.tokenizer.encode(input_text),
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)

            # Get attention weights
            self.model.eval()
            with torch.no_grad():
                attention_weights = self.model.get_attention_weights(input_ids)

            # Plot attention map
            if ax is None:
                plt.figure(figsize=self.figure_size)
                ax = plt.gca()

            attention_map = attention_weights[layer_idx][0, head_idx].cpu()

            # Create token labels
            tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]

            # Plot heatmap
            sns.heatmap(
                attention_map,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                ax=ax
            )

            # Set title and labels using correct method
            ax.set_title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})\nInput: "{input_text}"')
            ax.set_xlabel('Key Tokens')
            ax.set_ylabel('Query Tokens')

            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            if ax is None:
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_samples(self, prompts: List[str], max_tokens: int = 100, temperature: float = 1.0):
        """Generate text samples from given prompts."""
        print(f"\n✍️ Generating Text Samples (temperature={temperature})")

        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            try:
                # Tokenize without special token handling
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt),
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)

                generated_text = prompt
                print("\nGeneration:")
                print("-" * 40)

                for i in range(max_tokens):
                    # Get model predictions
                    with torch.no_grad():
                        logits, _ = self.model(input_ids)

                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :] / temperature
                    next_token_probs = F.softmax(next_token_logits, dim=-1)

                    # Sample next token
                    next_token = torch.multinomial(next_token_probs, num_samples=1)

                    # Decode new token (without special token handling)
                    new_token = self.tokenizer.decode([next_token.item()])
                    generated_text += new_token

                    # Print progress
                    print(f"Token {i+1:3d}: '{new_token}' ({next_token_probs[next_token].item():.4f})")

                    # Update input for next iteration
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    # Stop if end of text token is generated
                    if next_token.item() == self.tokenizer.encode("<|endoftext|>")[0]:
                        break

                print("\nFinal Text:")
                print("-" * 40)
                print(generated_text)
                print("-" * 40)

            except Exception as e:
                print(f"Error generating from prompt '{prompt}': {str(e)}")
                traceback.print_exc()
                continue

    def plot_training_progress(self):
        """
        Plot training and validation loss curves.
        Shows model's learning progress over time.
        """
        plt.figure(figsize=self.figure_size)
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_all_heads(self, input_text: str, layer_idx: int = 0):
        """Show attention patterns for all heads in a layer."""
        print(f"\n🔍 Visualizing All Attention Heads for Layer {layer_idx}")

        # Create single figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()  # Make it easier to iterate

        # Plot each head
        for head in range(self.model.n_heads):
            print(f"Plotting head {head}...")
            self.visualize_attention(input_text, layer_idx, head, ax=axes[head])

        plt.tight_layout()
        # Use non-blocking display
        plt.show(block=False)
        plt.pause(1)  # Give time for plot to render
        plt.close()

class GPTTrainer:
    def __init__(self, model, data_loader, evaluator):
        print("\n" + "="*80)
        print("🎓 Welcome to GPT-2 Training - A Beginner's Guide!")
        print("="*80)
        self.model = model
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self, epochs, batch_size, learning_rate):
        print("\n📚 Phase 1: Understanding Your Data")
        print("="*60)
        print(f"• Total text characters: {self.data_loader.total_chars:,}")
        print(f"• After tokenization: {self.data_loader.total_tokens:,} tokens")
        print("\n🤔 What are tokens?")
        print("   Tokens are the smallest units the model understands.")
        print("   Example: 'machine learning' → ['machine', 'learn', 'ing']")

        print("\n🧮 Data Split:")
        print(f"• Training: {self.data_loader.train_tokens:,} tokens")
        print(f"   └── Used to teach the model patterns")
        print(f"• Validation: {self.data_loader.val_tokens:,} tokens")
        print(f"   └── Used to check if model is really learning")

        print("\n🏗️ Phase 2: Model Architecture")
        print("="*60)
        print("The model is like a stack of smart filters:")
        print(f"• Input Layer: Converts {self.model.vocab_size:,} words → {self.model.d_model} numbers")
        print(f"• {self.model.n_layers} Transformer Layers:")
        print(f"   └── Each has {self.model.n_heads} different ways to look at text")
        print(f"   └── Can understand {self.model.block_size} words at once")

        print("\n🎯 Phase 3: Training Strategy")
        print("="*60)
        print(f"• Batch Size: {batch_size} sequences at once")
        print(f"   └── Like solving {batch_size} puzzles in parallel")
        print(f"• Learning Rate: {learning_rate}")
        print(f"   └── How big steps to take when learning")
        print(f"• Epochs: {epochs}")
        print(f"   └── Will see all data {epochs} times")

        for epoch in range(epochs):
            print(f"\n📈 Epoch {epoch+1}/{epochs}")
            print("-"*40)

            # Get batch
            inputs, targets = self.data_loader.get_batch()
            print("\n1️⃣ Processing Batch:")
            print(f"• Input shape: {inputs.shape}")
            print("   └── Each number represents a word/token")

            # Forward pass
            self.model.train()
            logits, loss = self.model(inputs, targets)
            print("\n2️⃣ Model Prediction:")
            print(f"• Made {logits.shape[-1]:,} predictions per position")
            print(f"• Current Loss: {loss.item():.4f}")
            print("   └── Lower is better (perfect = 0)")

            if epoch > 0:
                print(f"   └── Change: {loss.item() - prev_loss:+.4f}")
            prev_loss = loss.item()

            # Backward pass
            print("\n3️⃣ Learning Step:")
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient info
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            print(f"• Gradient Size: {grad_norm:.4f}")
            print("   └── How strongly the model wants to change")

            # Update weights
            old_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            new_lr = self.optimizer.param_groups[0]['lr']

            print("\n4️⃣ Model Update:")
            print(f"• Learning Rate: {old_lr:.6f} → {new_lr:.6f}")
            print("   └── Adjusting step size as we learn")

            # Evaluation
            if epoch % self.eval_interval == 0:
                val_loss = self.evaluator.evaluate_loss(self.data_loader)
                print("\n📊 Progress Check:")
                print(f"• Training Loss: {loss.item():.4f}")
                print(f"• Validation Loss: {val_loss:.4f}")
                print(f"• Perplexity: {math.exp(val_loss):.2f}")
                print("   └── How confused the model is (lower = better)")

                # Generate sample text
                if epoch > self.eval_interval:
                    print("\n✍️ Let's see what the model writes:")
                    self.evaluator.generate_samples(["The future of AI is"])

# 7. Create data loaders
train_loader = CustomDataLoader(training_data, training_batch_size, sequence_length)
eval_loader = CustomDataLoader(validation_data, evaluation_batch_size, sequence_length)

# 8. Initialize model
m = GPT(vocab_size=vocabulary_size, d_model=embedding_dimension, n_heads=attention_heads, n_layers=transformer_layers).to(device)
try:
    m = torch.compile(m)
    print("\n✅ Model successfully compiled")
except Exception as e:
    print(f"\n⚠️ Could not compile model: {str(e)}")

# 9. Initialize evaluator (after model is created)
evaluator = ModelEvaluator(model=m, tokenizer=tokenizer, device=device)

# 10. Training setup
optim = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=learning_rate*0.1)

# After model compilation
print("\n" + "="*80)
print("🚀 Training Configuration:")
print("="*80)
print(f"Training Tokens: {len(training_data):,}")
print(f"Batch Size: {training_batch_size}")
print(f"Total Batches per Epoch: {len(training_data) // (training_batch_size * sequence_length):,}")
print(f"Learning Rate: {learning_rate}")
print(f"Total Epochs: {total_epochs}")
print(f"Evaluation Interval: {evaluation_interval}")
print("="*80)

# Training Loop with Better Error Handling
print("\n🚀 Starting Training")
try:
    for epoch in range(total_epochs):
        # Initialize epoch timer
        epoch_start_time = time.time()

        print(f"\n📈 Epoch {epoch}/{total_epochs}")
        print("="*40)

        # Training step
        print("1️⃣ Getting batch...")
        input_ids, target_ids = train_loader.get_batch()
        print(f"   └── Input shape: {input_ids.shape}")
        print(f"   └── Target shape: {target_ids.shape}")

        # Forward pass
        print("\n2️⃣ Forward pass...")
        m.train()
        logits, current_loss = m(input_ids, target_ids)
        print(f"   └── Current loss: {current_loss.item():.4f}")

        # Backward pass
        print("\n3️⃣ Backward pass...")
        optim.zero_grad(set_to_none=True)
        current_loss.backward()

        # Gradient clipping
        print("\n4️⃣ Gradient clipping...")
        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1)
        print(f"   └── Gradient norm: {grad_norm:.4f}")

        # Optimizer step
        print("\n5️⃣ Optimizer step...")
        old_lr = optim.param_groups[0]['lr']
        optim.step()
        scheduler.step()
        new_lr = optim.param_groups[0]['lr']
        print(f"   └── Learning rate: {old_lr:.6f} → {new_lr:.6f}")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"\n⏱️ Epoch Stats:")
        print(f"   └── Time: {epoch_time:.2f}s")
        print(f"   └── Tokens/second: {(input_ids.shape[0] * input_ids.shape[1]) / epoch_time:.0f}")

        # Evaluation and visualization
        if epoch % evaluation_interval == 0:
            print(f"\n📊 Evaluation at Epoch {epoch}")
            print("="*40)

            # Evaluate on validation set
            validation_loss = evaluator.evaluate_loss(eval_loader)

            # Track losses
            evaluator.training_losses.append(current_loss.item())
            evaluator.validation_losses.append(validation_loss)

            print("\n📈 Training Progress:")
            print(f"   └── Training Loss: {current_loss.item():.4f}")
            print(f"   └── Validation Loss: {validation_loss:.4f}")
            print(f"   └── Perplexity: {math.exp(validation_loss):.2f}")

            # Plot training progress (non-blocking)
            evaluator.plot_training_progress()

            # Only do visualizations after some training
            if epoch > evaluation_interval * 2:
                if epoch % (evaluation_interval * 10) == 0:
                    print("\n🔍 Generating Attention Visualizations...")
                    evaluator.visualize_all_heads("The meaning of life is")

                if epoch % (evaluation_interval * 20) == 0:
                    print("\n✍️ Generating Sample Texts...")
                    prompts = [
                        "Once upon a time",
                        "The secret to happiness is",
                        "In the distant future"
                    ]
                    evaluator.generate_samples(prompts, temperature=0.8)

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    print("Stack trace:")
    import traceback
    traceback.print_exc()
finally:
    print("\n🏁 Training session ended")
    if 'current_loss' in locals():
        print(f"Final training loss: {current_loss.item():.4f}")
    else:
        print("No loss value available (training may have failed)")
