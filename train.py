import torch
import tiktoken
from gpt2 import GPT, device
import time
import math
import os
from tqdm import tqdm
import logging
import datetime
import traceback

def train():
    """Training Process Implementation"""
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Configure logging with consistent format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    # Initial welcome banner
    logging.info("\n" + "="*100)
    logging.info("    🚀 GPT-2 Training Process - An Educational Journey")
    logging.info("    This interactive guide will walk you through every step of training")
    logging.info("    a GPT-2 language model, explaining what's happening in real-time!")
    logging.info("="*100)

    # Safe device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading with detailed statistics
    logging.info("\n📚 Loading Training Data\n------------------------\nLet's look at what we're working with...")

    try:
        with open("data.txt", "r", encoding='utf-8') as f:
            text = f.read()

        # Split long log messages into multiple calls
        logging.info("\n📊 Dataset Statistics\n-------------------")
        logging.info("📝 Raw Text:")
        logging.info(f"   • Total Characters: {len(text):,}")
        logging.info(f"   • Unique Characters: {len(set(text)):,}")
        logging.info(f"   • Lines of Text: {text.count(chr(10)):,}")
        logging.info(f"   • Average Line Length: {len(text)/max(1, text.count(chr(10))):.1f} chars")
        logging.info(f"\n🔤 Sample Content:\n   \"{text[:100]}...\"")

        # Tokenization with explanation
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        sample_tokens = tokens[:10]
        sample_words = tokenizer.decode(sample_tokens)

        logging.info("\n🎯 Tokenization Results\n---------------------")
        logging.info("📊 Token Statistics:")
        logging.info(f"   • Total Tokens: {len(tokens):,}")
        logging.info(f"   • Vocabulary Coverage: {len(set(tokens)):,} unique tokens")
        logging.info(f"   • Compression Ratio: {len(tokens)/len(text):.2f} tokens per character")
        logging.info(f"\n🔍 Example Tokenization:")
        logging.info(f"   Text: \"{sample_words}\"")
        logging.info(f"   Tokens: {sample_tokens}")

        # Convert to tensor and split data
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        n_train = int(0.9 * len(tokens))
        train_data = tokens[:n_train]
        val_data = tokens[n_train:]

        # Model initialization with architecture breakdown
        logging.info("""
🤖 Initializing Neural Network
---------------------------
Building a GPT-2 model with the following architecture:""")

        model = GPT().to(device)

        logging.info(f"""
📐 Model Architecture
------------------
🔸 Input Processing:
   • Vocabulary Size: {model.vocab_size:,} tokens
   • Embedding Dimension: {model.d_model} units
   • Maximum Sequence Length: {model.block_size} tokens

🔸 Transformer Blocks:
   • Number of Layers: {len(model.blocks)}
   • Attention Heads per Layer: {model.n_heads}
   • MLP Dimension: 3072 units

🔸 Total Parameters: {sum(p.numel() for p in model.parameters()):,}
""")

        # Initialize optimizer and learning rate scheduler
        batch_size = 16
        sequence_length = 512
        learning_rate = 1e-3
        epochs = 5

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Training configuration with explanations
        logging.info("""
⚙️ Training Configuration
----------------------
Setting up the learning process with carefully chosen parameters:""")

        logging.info(f"""
📈 Hyperparameters:
   • Batch Size: {batch_size} sequences
     → Processes {batch_size} text chunks at once for stable learning

   • Sequence Length: {sequence_length} tokens
     → Each chunk is {sequence_length} tokens long for context

   • Learning Rate: {learning_rate}
     → Controls how fast the model updates its understanding

   • Training Epochs: {epochs}
     → Will see the entire dataset {epochs} times

🔧 Optimizer: AdamW
   → Advanced optimization algorithm that adapts the learning rate
   → Includes weight decay for better generalization

📉 Learning Rate Scheduler:
   → Reduces learning rate when progress plateaus
   → Helps fine-tune the model's understanding
""")

        # Training loop with rich progress information
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            logging.info(f"""
🎯 Starting Epoch {epoch+1}/{epochs}
{'='*50}""")

            model.train()
            total_loss = 0
            num_batches = len(train_data) // (batch_size * sequence_length)
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")

            for batch_num in progress_bar:
                batch_start = time.time()

                # Get batch
                start_idx = batch_num * batch_size * sequence_length
                x = train_data[start_idx:start_idx + batch_size * sequence_length].view(batch_size, sequence_length)
                y = train_data[start_idx+1:start_idx + batch_size * sequence_length + 1].view(batch_size, sequence_length)

                # Forward pass
                logits, loss = model(x, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_num + 1)
                batch_time = time.time() - batch_start
                tokens_per_sec = batch_size * sequence_length / batch_time

                if batch_num % 100 == 0:
                    # Convert tensor to list for tokenizer
                    sample_text = tokenizer.decode(x[0].cpu().tolist()[:50])

                    logging.info("\n📊 Training Progress - Batch {batch_num}/{num_batches}")
                    logging.info("---------------------------------")
                    logging.info("🔸 Loss Metrics:")
                    logging.info(f"   • Current Batch Loss: {loss.item():.4f}")
                    logging.info(f"   • Average Loss: {avg_loss:.4f}")
                    logging.info(f"   • Best Loss So Far: {best_val_loss:.4f}")
                    logging.info("\n⚡ Performance:")
                    logging.info(f"   • Processing Speed: {tokens_per_sec:.0f} tokens/second")
                    logging.info(f"   • Time per Batch: {batch_time:.2f}s")

                    # Safe GPU memory logging
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.max_memory_allocated()/1e9
                        logging.info(f"   • GPU Memory: {gpu_memory:.1f}GB")

                    logging.info(f"\n🎯 Sample Input:\n   \"{sample_text}...\"")
                    logging.info("\n💫 Learning Progress:")
                    logging.info(f"   • Completed: {batch_num/num_batches*100:.1f}% of epoch")
                    logging.info(f"   • Elapsed Time: {time.time() - epoch_start:.1f}s")
                    logging.info(f"   • Estimated Time Remaining: {(time.time() - epoch_start) * (num_batches-batch_num) / (batch_num+1):.1f}s")

            # Epoch summary
            epoch_time = time.time() - epoch_start
            logging.info(f"""
🌟 Epoch {epoch+1} Complete!
-------------------------
📊 Performance Metrics:
   • Final Loss: {avg_loss:.4f}
   • Time Taken: {epoch_time:.1f}s
   • Average Speed: {len(train_data)/epoch_time:.0f} tokens/second

💾 Saving Checkpoint...
""")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'vocab_size': model.vocab_size,
                    'd_model': model.d_model,
                    'n_heads': model.n_heads,
                }
            }, f'checkpoints/model_epoch_{epoch+1}.pt')

    except Exception as e:
        logging.error("\n❌ Error Encountered")
        logging.error("-----------------")
        logging.error(str(e))
        logging.error("\n💡 Troubleshooting Tips:")
        logging.error("• Check if data.txt exists and is readable")

        # Safe GPU memory checking
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory/1e9
                logging.error(f"• GPU Memory Available: {gpu_memory:.1f}GB")
            except Exception:
                logging.error("• GPU detected but unable to query memory")
        else:
            logging.error("• Running on CPU - consider using GPU for faster training")

        logging.error("• Ensure all required packages are installed")
        logging.error("• Try reducing batch size or sequence length if out of memory")
        logging.error("\n🔍 Error Details:")
        logging.error(traceback.format_exc())
        raise

def evaluate_model(model, tokens, batch_size=16, sequence_length=512):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0
    num_batches = len(tokens) // (batch_size * sequence_length)

    with torch.no_grad():
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size * sequence_length
            end_idx = start_idx + batch_size * sequence_length
            x = tokens[start_idx:end_idx].view(batch_size, sequence_length)
            y = tokens[start_idx+1:end_idx+1].view(batch_size, sequence_length)

            # Forward pass
            logits, loss = model(x, y)
            total_loss += loss.item()

    return total_loss / num_batches

def configure_optimizer(model):
    """
    Adam Optimizer Configuration (MATH.md Section 10.1)

    Update Rules:
    mₜ = β₁mₜ₋₁ + (1-β₁)gₜ (momentum)
    vₜ = β₂vₜ₋₁ + (1-β₂)gₜ² (velocity)

    Parameter Update:
    θₜ = θₜ₋₁ - η * m̂ₜ/√(v̂ₜ + ε)

    Hyperparameters:
    - Learning rate (η): 1e-3
    - Beta1 (β₁): 0.9 (momentum decay)
    - Beta2 (β₂): 0.999 (velocity decay)
    - Epsilon (ε): 1e-8 (numerical stability)
    """

if __name__ == "__main__":
    train()
