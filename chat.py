import torch
import tiktoken
from gpt2 import GPT
import time
import os
import sys
import logging
import datetime

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/chat_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_trained_model():
    """Load our trained GPT-2 model"""
    logger.info("Starting model loading process")
    print("\n" + "="*50)
    print("🤖 Loading Your Trained AI Model")
    print("="*50)

    # Check for available checkpoints
    checkpoints = sorted([f for f in os.listdir('.') if f.startswith('model_checkpoint')])

    if not checkpoints:
        print("❌ Error: No trained model found!")
        print("\nPlease train the model first:")
        print("1. Run: python train.py")
        print("2. Wait for training to complete")
        print("3. Try chat.py again")
        sys.exit(1)

    print("\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {cp}")

    checkpoint_path = checkpoints[-1]  # Use most recent
    print(f"\nUsing checkpoint: {checkpoint_path}")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model
        model = GPT().to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("✅ Model loaded successfully!")
        print(f"📊 Checkpoint info:")
        print(f"   Epoch: {checkpoint['epoch']+1}")
        print(f"   Loss: {checkpoint['loss']:.4f}")

        return model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

def chat():
    """Interactive chat with your trained GPT-2"""
    logger.info("Starting chat session")
    model = load_trained_model()
    enc = tiktoken.get_encoding("gpt2")

    # Get checkpoint info
    checkpoint = torch.load('model_checkpoint_final.pt', map_location=model.device)
    final_loss = checkpoint.get('loss', 'N/A')

    # Add model quality check with checkpoint loss
    logger.info(f"Model quality indicators:")
    logger.info(f"- Final loss: {final_loss:.4f}")
    logger.info(f"- Vocabulary size: {model.vocab_size:,} tokens")
    logger.info(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*50)
    print("🤖 Chat with Your AI")
    print("="*50)
    print("\nType your message (or 'quit' to exit)")
    print("Tips:")
    print("- Keep prompts clear and specific")
    print("- Type 'quit' to exit")
    print(f"- Current model loss: {final_loss:.4f}")

    while True:
        prompt = input("\n😊 You: ").strip()
        logger.info(f"User prompt: {prompt}")

        if prompt.lower() in ['quit', 'exit', 'q']:
            logger.info("Chat session ended by user")
            print("\n👋 Thanks for chatting!")
            break

        if prompt.lower() == 'debug':
            logger.setLevel(logging.DEBUG)
            print("\n🔍 Debug mode activated! You'll now see:")
            print("- Token probabilities")
            print("- Embedding activations")
            print("- Attention patterns")
            continue

        if prompt:
            try:
                print("\n🤖 AI: ", end="", flush=True)
                input_ids = torch.tensor(
                    enc.encode(prompt, disallowed_special=())
                ).unsqueeze(0).to(model.device)

                generated_text = ""

                with torch.no_grad():
                    for i in range(100):
                        logits, _ = model(input_ids)
                        next_token_logits = logits[0, -1, :] / 0.7
                        next_token_logits[next_token_logits < -1] = -float('Inf')

                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Get token info
                        token_id = next_token.item()
                        token_text = enc.decode([token_id])
                        token_prob = probs[token_id].item()

                        # Clear previous line and show token details
                        print(f"\r{'='*100}", flush=True)
                        print(f"\r🔄 Token #{i+1:3d}", end=' ', flush=True)
                        print(f"│ ID: {token_id:5d}", end=' ', flush=True)
                        print(f"│ Text: {token_text!r:<15}", end=' ', flush=True)
                        print(f"│ Prob: {token_prob:.3f}", end=' ', flush=True)

                        # Add confidence indicator
                        confidence = "⭐" if token_prob > 0.5 else "🤔" if token_prob > 0.1 else "❓"
                        print(f"│ Confidence: {confidence}", flush=True)

                        # Show token type
                        token_type = (
                            "WORD" if token_text.strip().isalnum() else
                            "PUNCT" if token_text in ",.!?-" else
                            "SPACE" if token_text.isspace() else
                            "SPECIAL"
                        )
                        print(f"│ Type: {token_type:8}", end=' ', flush=True)

                        # Show growing text on new line
                        if not token_text.isspace() or token_text == " ":
                            generated_text += token_text
                            print(f"\n📝 Full text: {generated_text}", flush=True)

                        # In debug mode, show top candidates
                        if logger.isEnabledFor(logging.DEBUG):
                            top_k = 5
                            topk_probs, topk_tokens = torch.topk(probs, top_k)
                            print("\n🎲 Top candidates:")
                            for prob, tok in zip(topk_probs, topk_tokens):
                                tok_text = enc.decode([tok.item()])
                                print(f"   {tok_text!r:<15} │ ID: {tok.item():5d} │ Prob: {prob:.3f}")

                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                        if token_id == enc.eot_token:
                            break

                        if torch.max(probs) < 0.05:
                            break

                        time.sleep(0.1)  # Slow down generation

                print("\n" + "="*100)  # Final separator
                print(f"\n🎯 Final generated text:\n{generated_text}\n")

            except KeyboardInterrupt:
                print("\n\nGeneration interrupted!")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Let's try again!")

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\n👋 Chat ended. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
