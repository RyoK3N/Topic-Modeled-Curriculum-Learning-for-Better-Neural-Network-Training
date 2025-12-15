# Topic-Modeled Curriculum Learning for Better Neural Network Training

## Overview

This repository (and research thesis) investigates a novel **Topic-Modeled Curriculum Learning (TMCL)** framework, where **topic modeling** is used to estimate the intrinsic difficulty of training samples and guide **curriculum learning** for neural networks.

Instead of relying on handcrafted heuristics or teacher models, this work proposes a **data-driven, unsupervised difficulty metric** derived from topic distributions. Training samples are introduced to the neural network in an *easy-to-hard* order based on semantic coherence, with the goal of improving **convergence speed, stability, and generalization**.

---

## Research problem statement

Deep neural networks are traditionally trained using **randomly shuffled datasets**, which ignores the internal semantic structure of data. While **curriculum learning (CL)** has shown that ordering samples from easy to hard can accelerate learning and improve performance, most existing CL approaches:

- Depend on handcrafted difficulty heuristics  
- Require teacher or auxiliary models  
- Do not scale well across domains  

This research proposes a **Topic-Modeled Curriculum Learning (TMCL)** framework, where **topic modeling** (LDA, NMF, Neural Topic Models) is used to:

1. Discover latent semantic structure in training data  
2. Quantify sample difficulty using topic distribution properties  
3. Construct an adaptive curriculum schedule for neural network training  

The central hypothesis is that **samples with concentrated topic distributions are easier to learn**, while **samples spanning multiple topics or noisy topic assignments are harder**.

---

## Some research questions

1. Can topic modeling provide a meaningful and scalable difficulty measure for training samples?
2. Does topic-modeled curriculum learning improve convergence speed and final performance?
3. How does TMCL compare with random sampling and traditional curriculum strategies?
4. Is the approach robust across datasets, tasks, and neural architectures?

---

## Hypotheses to be tested

### H1: Topic entropy reflects sample difficulty

Samples with **high topic entropy** are harder to learn than those with focused topic distributions.

### H2: Topic-Modeled curriculum improves training

Neural networks trained using TMCL will:
- Converge faster
- Achieve equal or better generalization
- Exhibit more stable loss curves

### H3: Cross-Domain robustness

TMCL benefits generalize across:
- Vision and NLP datasets
- CNNs and Transformer architectures

---

## Mathematical formulation

### Topic distribution

For a sample \( x_i \), a topic model produces a topic distribution:

\[
P(t \mid x_i), \quad t \in \{1, \dots, T\}
\]

---

### Difficulty score (Topic Entropy)

\[
D(x_i) = -\sum_{t=1}^{T} P(t \mid x_i)\log P(t \mid x_i)
\]

- Low entropy → Easy sample  
- High entropy → Hard sample  

---

### Curriculum scheduling function

\[
C(t) = D_{\min} + \frac{t}{T_{\max}}(D_{\max} - D_{\min})
\]

Where:
- \( t \): current training epoch  
- \( T_{\max} \): total epochs  
- \( D_{\min}, D_{\max} \): difficulty bounds  

Only samples with \( D(x_i) \le C(t) \) are eligible for training at epoch \( t \).

---

## Methodology

### 1. Topic modeling

#### Text data
- Preprocessing: tokenization, stop-word removal
- Topic Models:
  - LDA
  - NMF
  - Neural Topic Models (VAE-based)
- Output: per-document topic distribution

#### Vision data
- Extract embeddings using pretrained models (ResNet, ViT)
- Treat embeddings as pseudo-documents
- Apply soft clustering / topic modeling in embedding space

---

### 2. Curriculum construction

Samples are sorted by difficulty score and introduced progressively using the curriculum schedule.

---

### 3. Training regimes

| Regime          | Description                                     |
|-----------------|-------------------------------------------------|
| Random          | Standard shuffled mini-batches                  |
| Traditional CL  | Heuristic-based difficulty (length, complexity) |
| TMCL (Proposed) | Topic-modeled difficulty-based curriculum       |

---

## Models & architectures

| Dataset               | Model                          |
|-----------------------|--------------------------------|
| CIFAR-10 / CIFAR-100  | ResNet-18                      |
| MNIST / Fashion-MNIST | CNN                            |
| AG News / IMDb        | Transformer (BERT fine-tuning) |

---

## Evaluation metrics

### Performance metrics
- Accuracy / F1 Score
- Test loss

### Curriculum-specific metrics
- Convergence speed (epochs to threshold)
- Area under training loss curve
- Generalization gap

---

## Experiments

### Experiment 1: Difficulty validation
- Correlate topic entropy with early-epoch loss
- Validate entropy as a difficulty proxy

### Experiment 2: Curriculum effectiveness
- Compare convergence speed across regimes
- Analyze stability and loss smoothness

### Experiment 3: Cross-architecture robustness
- Repeat experiments across CNNs and Transformers

---

## Ablation studies

- Topic entropy vs. max-topic probability
- Linear vs. exponential curriculum pacing
- Effect of topic count \( T \)
- Fixed vs. adaptive curricula

---

## Related work

### Curriculum learning
- Bengio et al. (2009) – Curriculum Learning
- Soviany et al. (2021) – Survey of Curriculum Learning
- Hacohen & Weinshall (2019) – Self-paced learning
- Graves et al. (2017) – Automated curriculum learning

### Topic modeling
- Blei et al. – Latent Dirichlet Allocation
- Neural Topic Models (VAE-based)
- Topic modeling surveys and reviews

---

## Expected contributions

- A **novel unsupervised difficulty metric** based on topic entropy
- A **scalable curriculum learning framework**
- Empirical evidence across domains and architectures
- Practical guidelines for curriculum-driven training

---

## Future work

- Dynamic topic models for evolving curricula
- Multi-modal topic-modeled curricula
- Integration with large-scale language model training
- Curriculum learning for continual and lifelong learning

---

## Tools & libraries

- Python, PyTorch
- Gensim (LDA / NMF)
- HuggingFace Transformers
- Scikit-learn

---

## Author

**Reiyo**  
Research Topic: Topic-Modeled Curriculum Learning for better and efficient Neural Network Training
Field: Deep Learning / Representation Learning / Optimization
