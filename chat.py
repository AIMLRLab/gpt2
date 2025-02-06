import torch
import tiktoken
from gpt2 import GPT
import time
import os
import sys

def load_trained_model():
    """Load our trained GPT-2 model"""
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
    # Load model and tokenizer
    model = load_trained_model()
    enc = tiktoken.get_encoding("gpt2")

    print("\n" + "="*50)
    print("🤖 Chat with Your AI")
    print("="*50)
    print("\nType your message (or 'quit' to exit)")
    print("Tips:")
    print("- Keep prompts clear and specific")
    print("- Type 'quit' to exit")

    while True:
        prompt = input("\n😊 You: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for chatting!")
            break

        if prompt:
            try:
                print("\n🤖 AI: ", end="", flush=True)

                # Use the newer API for encoding
                input_ids = torch.tensor(
                    enc.encode(prompt, disallowed_special=())
                ).unsqueeze(0).to(model.device)

                with torch.no_grad():
                    for _ in range(100):
                        logits, _ = model(input_ids)
                        next_token_logits = logits[0, -1, :] / 0.7

                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                        # Simple decode without any special handling
                        token_text = enc.decode([next_token.item()])
                        print(token_text, end="", flush=True)

                        # Check for end token
                        if next_token.item() == enc.eot_token:
                            break

                print("\n")

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
