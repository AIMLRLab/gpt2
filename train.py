import torch
import tiktoken
from gpt2 import GPT, device
import time
import math
import os
from tqdm import tqdm
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def train():
    """
    Main training function for the GPT-2 model.

    This function handles:
    1. Data loading and preprocessing
    2. Model initialization
    3. Training loop with checkpoints
    4. Progress monitoring and logging
    5. Error handling and recovery
    """
    logging.info("\n" + "="*50)
    logging.info("🚀 Starting GPT-2 Training")
    logging.info("="*50)

    # Initialize tokenizer
    logging.info("\n📚 Initializing tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load and preprocess data
    logging.info("\n📖 Loading training data...")
    try:
        with open("data.txt", "r", encoding='utf-8') as f:
            text = f.read()
        logging.info(f"✅ Loaded {len(text):,} characters")

        # Convert text to tokens
        tokens = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long,
            device=device
        )
        logging.info(f"✅ Encoded to {len(tokens):,} tokens")

    except FileNotFoundError:
        logging.error("❌ Error: data.txt not found!")
        logging.info("\nPlease create a data.txt file with your training text.")
        return
    except Exception as e:
        logging.error(f"❌ Error reading data.txt: {str(e)}")
        return

    # Split data into train/val (90/10 split)
    split = int(0.9 * len(tokens))
    train_data = tokens[:split]
    val_data = tokens[split:]

    # Initialize model and optimizer
    logging.info("\n🤖 Initializing model...")
    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training parameters
    batch_size = 16
    sequence_length = 512
    epochs = 5

    # Log configuration
    logging.info("\n📊 Training Configuration:")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Sequence Length: {sequence_length}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Device: {device}")
    logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    try:
        # Training loop
        for epoch in range(epochs):
            logging.info(f"\n🔄 Epoch {epoch+1}/{epochs}")

            # Training
            model.train()
            total_loss = 0
            num_batches = len(tokens) // (batch_size * sequence_length)

            # Progress bar for batches
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")

            for i in progress_bar:
                # Get batch
                start_idx = i * batch_size * sequence_length
                end_idx = start_idx + batch_size * sequence_length
                x = tokens[start_idx:end_idx].view(batch_size, sequence_length)
                y = tokens[start_idx+1:end_idx+1].view(batch_size, sequence_length)

                # Forward pass
                logits, loss = model(x, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / num_batches
            logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            checkpoint = {
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
            }

            # Save both latest and epoch-specific checkpoints
            torch.save(checkpoint, 'model_checkpoint.pt')
            torch.save(checkpoint, f'model_checkpoint_epoch_{epoch+1}.pt')
            logging.info(f"✅ Saved checkpoints for epoch {epoch+1}")

            # Add to training loop
            val_loss = evaluate_model(model, tokens)
            logging.info(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, 'model_best.pt')

    except KeyboardInterrupt:
        logging.warning("\n⚠️ Training interrupted!")
        logging.info("Saving interrupted checkpoint...")
        torch.save(checkpoint, 'model_checkpoint_interrupted.pt')

    except Exception as e:
        logging.error(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Save final model
    try:
        torch.save(checkpoint, 'model_checkpoint_final.pt')
        logging.info("\n✅ Training complete! Final model saved.")
    except Exception as e:
        logging.error(f"\n❌ Error saving final model: {str(e)}")

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

if __name__ == "__main__":
    train()
