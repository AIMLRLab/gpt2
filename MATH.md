# Mathematical Foundations of GPT-2: From Ground Zero

## 📚 Glossary of Terms

- **Vector**: A list of numbers in a specific order (like coordinates on a map)
- **Matrix**: A grid or table of numbers (like a spreadsheet)
- **Dimension**: How many numbers are in a vector or how many rows/columns in a matrix
- **Scalar**: A single number
- **Function**: A rule that turns one number into another number
- **Parameter**: A number that can be adjusted or changed
- **Gradient**: The slope or steepness of a curve at a point
- **Optimization**: Finding the best possible solution (like finding the highest or lowest point)
- **Distribution**: A pattern that shows how often different values occur
- **Probability**: The chance of something happening (between 0 and 1)
- **Linear**: Following a straight line
- **Non-linear**: Not following a straight line (can be curved)
- **Embedding**: Converting words into lists of numbers
- **Token**: A piece of text (could be a word or part of a word)
- **Tensor**: Like a vector or matrix, but can have more dimensions
- **Norm**: The length or size of a vector
- **Convex**: Shaped like a bowl (curves upward)
- **Concatenation**: Joining things together end-to-end
- **Projection**: Converting data from one form to another
- **Initialization**: Setting starting values
- **Batch**: A group of examples processed together
- **Epoch**: One complete pass through all training data
- **Convergence**: When training starts to reach its best performance
- **Perplexity**: A measure of how well the model predicts text
- **Entropy**: A measure of uncertainty or randomness
- **Numerical Stability**: Preventing numbers from becoming too big or small
- **Exponential**: Growing very quickly (like doubling repeatedly)
- **Logarithm**: The opposite of exponential (like asking "2 to what power gives 8?")
- **Sinusoidal**: Following a wave pattern (like ocean waves)
- **Parallel**: Happening at the same time
- **Sequential**: Happening one after another
- **Variance**: How spread out numbers are from their average
- **Mean**: The average of a set of numbers

## 📑 Table of Contents

1. **Fundamental Concepts**

   - 1.1 What is a Vector?
   - 1.2 Vector Operations
     - 1.2.1 Vector Addition
     - 1.2.2 Dot Product
     - 1.2.3 Vector Magnitude
   - 1.3 What is a Matrix?

2. **Basic Neural Network Concepts**

   - 2.1 Linear Transformation
   - 2.2 Activation Functions
     - 2.2.1 ReLU
     - 2.2.2 GELU

3. **Probability and Statistics Fundamentals**

   - 3.1 Probability Distributions
     - 3.1.1 Softmax Function
   - 3.2 Cross-Entropy Loss

4. **Building Blocks of Attention**

   - 4.1 Self-Attention: The Core Idea
     - 4.1.1 Query, Key, Value Concept
     - 4.1.2 Basic Attention Formula
   - 4.2 Matrix Dimensions in Attention

5. **Positional Encoding Mathematics**

   - 5.1 The Need for Position Information
   - 5.2 Sinusoidal Position Encoding
     - 5.2.1 Formula
     - 5.2.2 Properties
     - 5.2.3 Mathematical Proof of Linear Relationship

6. **Multi-Head Attention in Detail**

   - 6.1 Single Head Mathematics
     - 6.1.1 Projection Matrices
     - 6.1.2 Attention Score Computation
   - 6.2 Multi-Head Mechanism
     - 6.2.1 Parallel Heads
     - 6.2.2 Concatenation and Projection

7. **Layer Normalization Mathematics**

   - 7.1 Statistical Normalization
     - 7.1.1 Mean and Variance
     - 7.1.2 Normalization Formula
   - 7.2 Learnable Parameters

8. **Complete Architecture Mathematics**

   - 8.1 Forward Pass Computation
     - 8.1.1 Input Processing
     - 8.1.2 Transformer Block
   - 8.2 Computational Complexity
     - 8.2.1 Attention Complexity
     - 8.2.2 Total Parameters

9. **Training Dynamics**

   - 9.1 Loss Landscape Analysis
     - 9.1.1 Loss Function Geometry
     - 9.1.2 Gradient Flow
   - 9.2 Learning Rate Scheduling
     - 9.2.1 Cosine Schedule with Warmup
     - 9.2.2 Linear Schedule with Warmup
   - 9.3 Gradient Clipping
     - 9.3.1 Global Norm Computation
     - 9.3.2 Clipping Formula

10. **Optimization Theory**

    - 10.1 Adam Optimizer Mathematics
      - 10.1.1 Update Rules
    - 10.2 Weight Initialization
      - 10.2.1 Kaiming Initialization
      - 10.2.2 Xavier/Glorot Initialization
    - 10.3 Batch Processing Mathematics
      - 10.3.1 Mini-batch Statistics
      - 10.3.2 Gradient Estimation
    - 10.4 Convergence Analysis
      - 10.4.1 Expected Learning Progress
      - 10.4.2 Training Stability Conditions

11. **Model Evaluation Metrics**

    - 11.1 Perplexity
    - 11.2 Cross-Entropy Loss
    - 11.3 KL Divergence
    - 11.4 Gradient Norms

12. **Numerical Stability**
    - 12.1 Log-Space Computations
    - 12.2 Gradient Scaling
    - 12.3 Mixed Precision Training

## 1. Fundamental Concepts

### 1.1 What is a Vector?

A vector is an ordered list of numbers. In machine learning, we use vectors to represent things mathematically.

Example: Word "hello" might be represented as: v = [0.2, -0.5, 0.1, 0.8, -0.3]

Think about it this way: When you describe the weather, you might use multiple numbers - temperature (75°), humidity (50%), chance of rain (20%). Together, these numbers form a vector that describes the weather: [75, 50, 20].

Let's build up to the formal definition:

1. A single number is like a point on a line
2. Two numbers can represent a point on a map (like coordinates)
3. Three numbers can represent a point in space
4. More numbers? Same idea, just more dimensions!

Formally: A vector x ∈ ℝⁿ is written as:
x = [x₁, x₂, ..., xₙ]

### 1.2 Vector Operations

#### 1.2.1 Vector Addition

When we add vectors, we add the corresponding numbers in each position.

Example:
Weather today: [75°, 50%, 20%]
Change expected: [+5°, -10%, +30%]
Tomorrow's forecast: [80°, 40%, 50%]

The math: For vectors a = [a₁, a₂, ..., aₙ] and b = [b₁, b₂, ..., bₙ]:
a + b = [a₁ + b₁, a₂ + b₂, ..., aₙ + bₙ]

#### 1.2.2 Dot Product

The dot product is a way to multiply vectors that gives us a single number. It tells us how similar two vectors are.

Real-world example:
If you have monthly expenses [rent, food, utilities] = [1000, 300, 200]
And time spent thinking about each [2, 1, 0.5] hours
Dot product = (1000×2) + (300×1) + (200×0.5) = 2300

This tells us how much these expenses weigh on your mind in terms of both time and money.

The math: a · b = ∑ᵢ(aᵢ × bᵢ) = a₁b₁ + a₂b₂ + ... + aₙbₙ

#### 1.2.3 Vector Magnitude

||x|| = √(∑ᵢxᵢ²)

Think of magnitude as measuring "how big" a vector is. If you're walking, your velocity vector [3, 4] (3 units east, 4 units north) has a magnitude of 5 units per hour (using the Pythagorean theorem).

### 1.3 What is a Matrix?

A matrix is a 2D array of numbers. We use matrices to transform vectors.

Example:

```
A = [
[1, 2, 3],
[4, 5, 6]
]
```

Think of a matrix like a transformation machine. When you multiply a vector by a matrix, it's like putting the vector through a machine that stretches, rotates, or squishes it in specific ways.

Formally: A matrix A ∈ ℝᵐˣⁿ has m rows and n columns.

## 2. Basic Neural Network Concepts

### 2.1 Linear Transformation

The most basic operation in neural networks is the linear transformation:
y = Wx + b

Where:

- x is input vector (n×1)
- W is weight matrix (m×n)
- b is bias vector (m×1)
- y is output vector (m×1)

Imagine you're converting temperatures from Celsius to Fahrenheit. The formula is F = 1.8C + 32. This is a linear transformation where:

- W (weight) is 1.8
- b (bias) is 32
- x (input) is the Celsius temperature
- y (output) is the Fahrenheit temperature

### 2.2 Activation Functions

#### 2.2.1 ReLU (Rectified Linear Unit)

ReLU(x) = max(0, x)

Think of ReLU as a bouncer at a club: if a number is negative, it gets turned away (set to 0). If it's positive, it gets to pass through unchanged.

Why we need it: Without ReLU or similar functions, neural networks could only learn linear patterns. ReLU lets them learn complex, curved patterns.

#### 2.2.2 GELU (Gaussian Error Linear Unit)

GELU(x) = x × P(X ≤ x)
where P(X ≤ x) is the cumulative distribution function of the standard normal distribution.

GELU is like a smarter bouncer. Instead of a strict "no negatives" policy, it gradually reduces the impact of negative numbers and slightly adjusts positive ones. This often works better than ReLU for language tasks.

Approximation:
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

## 3. Probability and Statistics Fundamentals

### 3.1 Probability Distributions

#### 3.1.1 Softmax Function

Converts raw scores into probabilities:

softmax(x)ᵢ = exp(xᵢ)/∑ⱼexp(xⱼ)

Imagine you're rating three movies with scores [4, 2, 0]. Softmax turns these into probabilities that add up to 1, showing how likely you are to choose each movie:

1. First, calculate e⁴ = 54.6, e² = 7.4, e⁰ = 1
2. Sum them: 54.6 + 7.4 + 1 = 63
3. Divide each by the sum: [0.87, 0.12, 0.01]

Properties:

1. Output range: [0,1]
2. Sum of outputs = 1
3. Preserves relative ordering

### 3.2 Cross-Entropy Loss

H(p,q) = -∑ᵢ p(i)log(q(i))

Cross-entropy measures how different two probability distributions are. Think of it as grading the model's predictions:

- Perfect prediction = 0 loss
- Terrible prediction = High loss

Example:
True distribution: p = [1, 0, 0] (correct answer is first option)
Model prediction: q = [0.8, 0.1, 0.1]
Loss = -(1×log(0.8) + 0×log(0.1) + 0×log(0.1)) = 0.223

## 4. Building Blocks of Attention

### 4.1 Self-Attention: The Core Idea

#### 4.1.1 Query, Key, Value Concept

Think of attention like looking up information in a database:

- Query (Q): What you're looking for
- Key (K): What you match against
- Value (V): What you get back

Real-world example: When you search for a video online:

- Query: Your search terms
- Keys: Video titles and descriptions
- Values: The actual videos
- Attention scores: How well each video matches your search

#### 4.1.2 Basic Attention Formula

Attention(Q,K,V) = softmax(QKᵀ/√d)V

Step by step:

1. Compute similarity scores: QKᵀ
2. Scale scores: QKᵀ/√d (prevents numbers from getting too big)
3. Convert to probabilities: softmax(QKᵀ/√d)
4. Get weighted values: (softmax(QKᵀ/√d))V

Example:
If you're reading "The cat sat on the mat", when processing "sat", the model pays attention to "cat" (subject) and "mat" (location) more than "the".

### 4.2 Matrix Dimensions in Attention

For sequence length n and embedding dimension d:

- Q ∈ ℝⁿˣᵈ (n queries, each d-dimensional)
- K ∈ ℝⁿˣᵈ (n keys, each d-dimensional)
- V ∈ ℝⁿˣᵈ (n values, each d-dimensional)
- Output ∈ ℝⁿˣᵈ (n outputs, each d-dimensional)

## 5. Positional Encoding Mathematics

### 5.1 The Need for Position Information

Unlike humans reading left-to-right, transformers process all words at once. Position encodings tell the model where each word is in the sentence.

Example: "Dog bites man" means something very different from "Man bites dog"

### 5.2 Sinusoidal Position Encoding

#### 5.2.1 Formula

For position pos and dimension i:
PE(pos,2i) = sin(pos/10000^(2i/d))
PE(pos,2i+1) = cos(pos/10000^(2i/d))

Think of this like a musical chord: different frequencies combine to create a unique pattern for each position.

#### 5.2.2 Properties

1. Unique for each position (like giving each word slot a special ID)
2. Bounded between [-1, 1] (keeps numbers manageable)
3. Linear relationship between positions (helps model understand relative distances)

Example:
Position 0: [sin(0), cos(0), sin(0/100), cos(0/100), ...]
Position 1: [sin(1), cos(1), sin(1/100), cos(1/100), ...]

Each position gets a unique "fingerprint" that the model can recognize.

## 6. Multi-Head Attention in Detail

### 6.1 Single Head Mathematics

#### 6.1.1 Projection Matrices

Think of these as different "viewpoints" or "perspectives" on the same information:

For embedding dimension d and head dimension dₕ:

- Wᵠ ∈ ℝᵈˣᵈʰ (Query projection)
- Wᵏ ∈ ℝᵈˣᵈʰ (Key projection)
- Wᵛ ∈ ℝᵈˣᵈʰ (Value projection)

Example:
When you look at a painting, you might focus on:

1. Colors (one perspective)
2. Shapes (another perspective)
3. Textures (yet another perspective)

#### 6.1.2 Attention Score Computation

1. Project input X ∈ ℝⁿˣᵈ:

   - Q = XWᵠ (what we're looking for)
   - K = XWᵏ (what we match against)
   - V = XWᵛ (what we get back)

2. Compute attention weights:
   A = softmax(QKᵀ/√dₕ)

3. Get output:
   O = AV

### 6.2 Multi-Head Mechanism

#### 6.2.1 Parallel Heads

Just like having multiple experts look at the same problem from different angles:

For h heads:
headᵢ = Attention(XWᵠᵢ, XWᵏᵢ, XWᵛᵢ)

#### 6.2.2 Concatenation and Projection

MHA(X) = Concat(head₁,...,headₕ)Wᵒ

Where:

- Concat(head₁,...,headₕ) ∈ ℝⁿˣ(ʰᵈʰ)
- Wᵒ ∈ ℝ(ʰᵈʰ)ˣᵈ

Think of this like getting opinions from multiple experts (heads), then combining their advice into one final decision.

## 7. Layer Normalization Mathematics

### 7.1 Statistical Normalization

#### 7.1.1 Mean and Variance

For input x ∈ ℝᵈ:
μ = (1/d)∑ᵢxᵢ
σ² = (1/d)∑ᵢ(xᵢ - μ)²

Think of this like adjusting the brightness and contrast of a photo:

- Mean (μ) is like the overall brightness level
- Variance (σ²) is like the contrast between light and dark areas

#### 7.1.2 Normalization Formula

x̂ = (x - μ)/√(σ² + ε)

Where ε is small constant (typically 1e-5) for numerical stability

Example:
If you have test scores [60, 70, 90]:

1. Calculate mean: μ = (60 + 70 + 90)/3 = 73.33
2. Calculate variance: σ² = ((60-73.33)² + (70-73.33)² + (90-73.33)²)/3 = 177.78
3. Normalize: [60-73.33, 70-73.33, 90-73.33]/√177.78 = [-1, -0.25, 1.25]

### 7.2 Learnable Parameters

y = γx̂ + β

Where:

- γ ∈ ℝᵈ (scale parameter)
- β ∈ ℝᵈ (shift parameter)

These parameters let the model adjust the normalization for each layer, like fine-tuning the brightness and contrast for each part of an image.

## 8. Complete Architecture Mathematics

### 8.1 Forward Pass Computation

#### 8.1.1 Input Processing

1. Token embeddings: E = XWᵉ
2. Position embeddings: P = pos_encode(positions)
3. Combined: H₀ = E + P

Example:
When processing "Hello world":

1. Convert each word to numbers (token embeddings)
2. Add position information (position embeddings)
3. Combine them to get the initial representation

#### 8.1.2 Transformer Block

For each layer l:

1. Multi-head attention:
   Ĥₗ = LN(Hₗ₋₁ + MHA(Hₗ₋₁))

2. Feed-forward:
   Hₗ = LN(Ĥₗ + FFN(Ĥₗ))

Where FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

Think of each transformer block like a station in an assembly line:

1. First station (attention): Look at all parts of the input
2. Second station (feed-forward): Process each position independently
3. Quality control (layer norm): Make sure everything's in the right range

### 8.2 Computational Complexity

#### 8.2.1 Attention Complexity

Time: O(n²d)
Memory: O(n²)

Where:

- n is sequence length
- d is embedding dimension

Why quadratic complexity?

- For each word (n)
- We compare it to every other word (n)
- Using vectors of size d

Example:
For a 1000-word text:

- Need to compute 1,000,000 attention scores
- Each score involves vectors of size d
- Total operations ≈ 1,000,000 × d

#### 8.2.2 Total Parameters

For model with L layers:
P = L(4d² + 8d²) + 2nd

Where:

- First term: transformer blocks
- Second term: embeddings
- n is vocabulary size

Example:
For a model with:

- 12 layers (L=12)
- 768 dimensions (d=768)
- 50,000 vocabulary size (n=50,000)
  Total parameters ≈ 124 million

## 9. Training Dynamics

### 9.1 Loss Landscape Analysis

#### 9.1.1 Loss Function Geometry

The loss surface L(θ) : ℝᵈ → ℝ has properties:

1. Non-convex (has many hills and valleys)
2. High-dimensional (d ≈ 124M in our implementation)
3. Contains multiple local minima (many "good enough" solutions)

Think of it like a mountain range:

- You want to find the lowest point (minimum)
- There are many valleys (local minima)
- Some paths down are steeper than others (gradients)

#### 9.1.2 Gradient Flow

∇L(θ) = [∂L/∂θ₁, ..., ∂L/∂θₙ]

For each parameter θᵢ:
∂L/∂θᵢ = ∑ₜ(∂L/∂yₜ)(∂yₜ/∂θᵢ)

This is like having a compass that points downhill in many directions at once. Each direction represents how changing one parameter affects the overall error.

### 9.2 Learning Rate Scheduling

#### 9.2.1 Cosine Schedule with Warmup

η(t) = η_max _ (0.5 + 0.5 _ cos(πt/T))

Warmup phase:
η(t) = η_max \* (t/t_warmup) for t < t_warmup

Think of this like learning to drive:

1. Start slow (warmup)
2. Gradually increase speed
3. Smoothly slow down near the end

#### 9.2.2 Linear Schedule with Warmup

η(t) = {
η_max _ (t/t_warmup) if t < t_warmup
η_max _ (1 - (t-t_warmup)/(T-t_warmup)) otherwise
}

This is like:

1. Walking at first (warmup)
2. Running at full speed
3. Gradually slowing down

### 9.3 Gradient Clipping

#### 9.3.1 Global Norm Computation

||g|| = √(∑ᵢ||gᵢ||²)

This measures how big all the gradients are together, like measuring the total force of many small pushes.

#### 9.3.2 Clipping Formula

g_clipped = g \* min(1, clip_norm/||g||)

Think of this like having a speed limit:

- If you're going too fast (gradient too large), slow down
- If you're under the limit, maintain your speed

## 10. Optimization Theory

### 10.1 Adam Optimizer Mathematics

#### 10.1.1 Update Rules

For each parameter θ:

mₜ = β₁mₜ₋₁ + (1-β₁)gₜ (First moment - average direction)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ² (Second moment - average speed)

Bias correction:
m̂ₜ = mₜ/(1-β₁ᵗ)
v̂ₜ = vₜ/(1-β₂ᵗ)

Parameter update:
θₜ = θₜ₋₁ - η \* m̂ₜ/√(v̂ₜ + ε)

Think of this like a ball rolling down a hill:

- First moment (mₜ) is like momentum - which way it's rolling
- Second moment (vₜ) is like friction - how much to slow down
- The update is like taking a step in the best direction

### 10.2 Weight Initialization

#### 10.2.1 Kaiming Initialization

For layer with n_in inputs:
W ~ N(0, √(2/n_in))

This is like dealing cards:

- Start with random values
- Scale them based on how many there are
- This prevents signals from getting too big or too small

#### 10.2.2 Xavier/Glorot Initialization

W ~ N(0, √(2/(n_in + n_out)))

Similar to Kaiming, but considers both input and output sizes.

### 10.3 Batch Processing Mathematics

#### 10.3.1 Mini-batch Statistics

For batch B:
μᵦ = (1/|B|)∑ᵢxᵢ
σ²ᵦ = (1/|B|)∑ᵢ(xᵢ - μᵦ)²

Like taking the average and spread of a sample instead of the whole population.

#### 10.3.2 Gradient Estimation

∇L ≈ (1/|B|)∑ᵢ∇Lᵢ

This is like:

- Taking a small sample of a large population
- Using that sample to guess the properties of the whole

### 10.4 Convergence Analysis

#### 10.4.1 Expected Learning Progress

For step size η:
E[||θₜ₊₁ - θ*||²] ≤ (1 - ηµ)||θₜ - θ\*||² + η²G²/(2µ)

This formula tells us:

- How quickly we can expect to learn
- What might slow us down
- When we might be done

#### 10.4.2 Training Stability Conditions

1. Gradient norm bound: ||∇L(θ)|| ≤ G
2. Lipschitz continuity: ||∇L(θ₁) - ∇L(θ₂)|| ≤ L||θ₁ - θ₂||
3. Learning rate bound: η ≤ 2/L

These are like safety rules:

- Don't take steps that are too big
- Make sure the landscape is smooth enough
- Keep your speed under control

## 11. Model Evaluation Metrics

### 11.1 Perplexity

PPL = exp(-1/N ∑ᵢlog P(xᵢ|x<ᵢ))

### 11.2 Cross-Entropy Loss

H(p,q) = -∑ᵢp(i)log(q(i))

### 11.3 KL Divergence

D_KL(P||Q) = ∑ᵢp(i)log(p(i)/q(i))

### 11.4 Gradient Norms

||∇L||₂ = √(∑ᵢ(∂L/∂θᵢ)²)

## 12. Numerical Stability

### 12.1 Log-Space Computations

For softmax:
log_softmax(x) = x - log(∑exp(x))

### 12.2 Gradient Scaling

For deep networks:
∂L/∂θₗ = ∏ᵢ₌ₗᴺ(∂hᵢ₊₁/∂hᵢ)(∂L/∂hₙ)

### 12.3 Mixed Precision Training

FP32 master weights: θ_master
FP16 computations: θ_compute = cast_fp16(θ_master)

[End of Mathematical Foundations]
