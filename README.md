# Build GPT-2 from Scratch

A PyTorch implementation of GPT-2 built from the ground up for learning purposes. This project demonstrates how to create a smaller version of OpenAI's GPT-2 language model, with step-by-step implementation details.

## Overview

This project breaks down the GPT-2 architecture into digestible components, allowing you to understand and build a language model from scratch. We'll implement:

- Custom tokenizer
- Data loader
- Basic language model
- Full GPT-2 architecture

## Project Structure

.
├── README.md
├── data/
│ └── data.txt # Training data (Taylor Swift & Ed Sheeran lyrics)
├── src/
│ ├── tokenizer.py # Custom character-level tokenizer
│ ├── dataloader.py # Data loading and batching
│ ├── model.py # GPT-2 model implementation
│ └── train.py # Training loop and utilities

## Requirements

- Python 3.8+
- PyTorch
- NumPy

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository
2. Install dependencies
3. Add your training data to `data/data.txt`
4. Run the training script:

```bash
python src/train.py
```

## Model Architecture

The implementation follows the original GPT-2 architecture with some simplifications:

- Character-level tokenization
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

## Training

The model is trained to predict the next character in a sequence. Training parameters:

- Batch size: 16
- Context length: 256
- Learning rate: 0.001
- Optimizer: AdamW

## Usage

```python
from src.model import GPT
from src.tokenizer import encode, decode

# Initialize model
model = GPT(vocab_size=vocab_size, d_model=d_model)

# Generate text
input_text = "Your prompt here"
tokens = encode(input_text)
output = model.generate(tokens, max_new_tokens=100)
generated_text = decode(output)
```

## References

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar
- Original GPT-2 paper: ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
