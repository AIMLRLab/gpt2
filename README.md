# 🤖 Mini GPT-2 Implementation

A beginner-friendly implementation of GPT-2 for learning purposes.

## 🎯 What Are We Building?

Imagine having an AI friend who can:

- ✍️ Continue writing stories you start
- 🎵 Generate song lyrics
- 📝 Help with creative writing
- 🤔 Answer questions about topics it's learned

## 🎓 Learning Journey

### 1️⃣ The Building Blocks

```python
# Each word becomes numbers the AI can understand
"Hello world!" → [3748, 995, 0]
```

- **Tokenizer**: Turns text into numbers (and back!)
- **Attention**: Helps AI understand which words are related
- **Neural Network**: The AI's "brain" for processing information

### 2️⃣ How It Works

Think of it like this:

1. 📚 AI reads lots of text (like learning from books)
2. 🧩 Breaks text into small pieces (tokens)
3. 🔍 Learns patterns (like how words go together)
4. ✍️ Uses patterns to write new text

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/AIMLRLab/gpt2.git
cd gpt2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Train the model
python gpt2.py

# Chat with the trained model
python chat.py
```

## 📊 Model Architecture

- Vocabulary Size: 50,257 tokens
- Embedding Dimension: 768
- Attention Heads: 4
- Transformer Layers: 8
- Context Length: 512 tokens

## 🎯 Training Details

- Batch Size: 16
- Learning Rate: 0.001
- Epochs: 5
- Optimizer: AdamW
- Loss: Cross Entropy

## 🎯 Usage

After training, chat with your model:

```bash
python chat.py
```

Example prompts:

- "Once upon a time"
- "The future of AI is"
- "The meaning of life is"

## 📊 Model Details

### Architecture (The AI's Brain)

```
Input → Embeddings → Transformer Blocks → Output
      ↑          ↑                    ↑
Words to    Position     12 layers of smart
numbers     info         pattern recognition
```

### Training Settings

- **Batch Size**: 16 (processes 16 text chunks at once)
- **Context Length**: 512 (can "remember" 512 tokens)
- **Learning Rate**: 0.001 (how fast it learns)
- **Model Size**: 124M parameters (like 124M knobs to tune)

### What These Numbers Mean

- **Batch Size**: Like solving 16 math problems at once
- **Context Length**: How many words it reads at once (like your short-term memory)
- **Learning Rate**: How big steps it takes when learning (too big = stumble, too small = slow)
- **Model Size**: How many "brain cells" it has (more = smarter but slower)

## 💡 Example Usage

```python
from gpt2 import GPT2

# Create AI model
model = GPT2()

# Generate text
prompt = "Once upon a time"
generated_text = model.generate(prompt, max_length=100)
print(generated_text)
```

## 🎮 Playground

Try these prompts:

1. "The future of AI is"
2. "Once upon a time"
3. "The secret to happiness"

## 📚 Learning Resources

Want to learn more? Check out:

- 🔗 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- 📖 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 🎥 [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0)

## ⚠️ Limitations

Remember:

- Needs lots of training data
- Can make mistakes
- Learns patterns but doesn't truly "understand"
- Works best with topics it's trained on

## 🤝 Contributing

Join our learning journey!

1. Fork [the repository](https://github.com/AIMLRLab/gpt2)
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Add your improvements
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📝 License

MIT License

## 🙋‍♂️ Questions?

- [Open an issue](https://github.com/AIMLRLab/gpt2/issues)
- Start a [discussion](https://github.com/AIMLRLab/gpt2/discussions)
- Join our Discord community

Remember: The goal is learning - don't be afraid to experiment and make mistakes!

## ⚠️ Known Issues

1. Training might seem to hang after "Training Progress" - this is normal, it's generating visualizations
2. First generation might be slow as the model loads
3. GPU recommended for faster training
