# Topic-Modeled Curriculum Learning for Better Neural Network Training

## Overview

This repository implements and documents research on **Topic-Modeled Curriculum Learning (TMCL)**, a novel framework that leverages **topic modeling** to estimate the intrinsic difficulty of training samples and guide **curriculum learning** for neural networks. Unlike traditional curriculum learning methods that rely on handcrafted heuristics or auxiliary models, TMCL provides a **data-driven, unsupervised difficulty metric** derived from the latent semantic structure of the data. The core principle is to train neural networks in an *easy-to-hard* order based on topic coherence, aiming to improve **convergence speed, training stability, and final generalization performance**.

---

## Research Problem Statement

Deep neural networks are predominantly trained on **randomly shuffled datasets**, which treats all samples as equally challenging and ignores the inherent semantic structure and varying complexity within the data. While **Curriculum Learning (CL)**—the idea of presenting samples in a meaningful order—has demonstrated benefits in accelerating learning and enhancing model robustness, most existing CL approaches suffer from significant limitations:

- **Heuristic Dependency:** Rely on manually defined difficulty measures (e.g., sentence length, image sharpness) that are domain-specific and often poor proxies for true learning complexity.
- **Scalability Issues:** Methods requiring pre-trained teacher models or reinforcement learning for curriculum generation are computationally expensive and difficult to scale.
- **Lack of Generalization:** Many strategies are tailored to specific tasks (e.g., NLP or Vision) and do not transfer well across domains.

This research addresses these gaps by proposing **Topic-Modeled Curriculum Learning (TMCL)**. TMCL uses **topic modeling** to:
1. Automatically discover the latent thematic structure within a training corpus.
2. Quantify sample difficulty through statistical properties of its topic distribution.
3. Construct an adaptive, semantically-informed curriculum schedule for neural network training.

The central hypothesis is that a sample's semantic "focus" or "purity," as captured by its topic distribution, correlates with its learnability. **Samples with concentrated, low-entropy topic distributions are semantically coherent and "easier" to learn, while samples with high-entropy, dispersed distributions are semantically ambiguous or complex and thus "harder."**

---

## Core Research Questions

1.  **Difficulty Metric:** Can topic modeling provide a meaningful, scalable, and unsupervised measure of sample difficulty that correlates with actual training dynamics (e.g., loss convergence)?
2.  **Training Efficacy:** Does a curriculum structured by topic-modeled difficulty lead to faster convergence, lower final error, and improved generalization compared to standard random sampling?
3.  **Comparative Performance:** How does TMCL perform against established curriculum learning baselines (e.g., length-based, loss-based) and self-paced learning?
4.  **Generalization & Robustness:** Is the TMCL framework effective across diverse domains (NLP, Vision), tasks (classification, regression), and model architectures (CNNs, Transformers)?

---

## Hypotheses and Theoretical Foundation

### H1: Topic Entropy Reflects Sample Difficulty
The entropy of a sample's topic distribution is a valid proxy for its learning complexity.
*   **Mathematical Formulation:** For a sample \( x_i \) with a topic distribution \( P(t | x_i) \) over \( T \) topics, its difficulty score is the Shannon entropy:
    \[
    D_{\text{entropy}}(x_i) = H(P) = -\sum_{t=1}^{T} P(t | x_i) \log P(t | x_i)
    \]
    We hypothesize that \( D_{\text{entropy}} \) will be positively correlated with the sample's cross-entropy loss in the early stages of model training.

### H2: Topic-Modeled Curriculum Improves Training Dynamics
Neural networks trained with a TMCL schedule will exhibit superior training characteristics.
*   **Convergence Speed:** Models will reach a target performance threshold in fewer epochs.
    *   *Metric:* Epochs to reach \( \alpha\% \) of final accuracy (\( \alpha \in \{90, 95\} \)).
*   **Generalization:** Models will achieve lower final test error and a smaller generalization gap.
    *   *Metric:* \( \text{Generalization Gap} = \mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}} \).
*   **Training Stability:** Loss curves will be smoother with lower variance between training runs.
    *   *Metric:* Variance of training loss across epochs \( \sigma^2(\mathcal{L}_{\text{train}}) \).

### H3: Cross-Domain Robustness
The benefits of TMCL are not architecture or domain-specific and will generalize across:
*   **Domains:** Text (AG News, IMDb) and Image (CIFAR-10/100, MNIST) datasets.
*   **Architectures:** Convolutional Neural Networks (ResNet) and Transformer-based models (BERT).

---

## Extended Mathematical Framework

### 1. Topic Modeling & Difficulty Scoring
Let a dataset be 𝒟 = {x₁, x₂, ..., xₙ}. A topic model (e.g., LDA, NMF) learns T topics, each represented as a distribution over features or words. For each sample xᵢ, the model infers a topic distribution:  
P(t | xᵢ) = [pᵢ₁, pᵢ₂, ..., pᵢₜ], where Σₜ pᵢₜ = 1.

**Alternative Difficulty Metrics (Ablations):**
- **Max Probability (Purity):**  
  Dₘₐₓ(xᵢ) = 1 − maxₜ P(t | xᵢ)  
  (Lower max probability ⇒ higher semantic ambiguity)
- **Topic Coherence Deviation:**  
  For samples with ground-truth label yᵢ, compute the average topic distribution for class k: P̄ₖ.  
  Difficulty = 1 − cosine_similarity(P(t | xᵢ), P̄_{yᵢ})
- **Composite Score:**  
  D꜀ₒₘₚ(xᵢ) = λ ⋅ H(P) + (1 − λ) ⋅ Dₘₐₓ(xᵢ), where λ ∈ [0, 1]  
  and H(P) = −Σₜ P(t | xᵢ) log P(t | xᵢ) (Shannon entropy)

### 2. Curriculum Scheduling Function
The curriculum defines a difficulty threshold τ(e) at epoch e. Only samples with D(xᵢ) ≤ τ(e) are used.

- **Linear Schedule:**  
  τₗᵢₙ(e) = Dₘᵢₙ + (e / E) ⋅ (Dₘₐₓ − Dₘᵢₙ)  
  where E = total epochs, Dₘᵢₙ/Dₘₐₓ = min/max difficulty in 𝒟
- **Root Schedule (Slow Start):**  
  τᵣₒₒₜ(e) = Dₘᵢₙ + (e / E)ᵞ ⋅ (Dₘₐₓ − Dₘᵢₙ), with γ < 1
- **Exponential Schedule (Fast Start):**  
  τₑₓₚ(e) = Dₘₐₓ − (Dₘₐₓ − Dₘᵢₙ) ⋅ βᵉ, with β ∈ (0, 1)

The proportion of data used at epoch e is:  
ρ(e) = |{xᵢ : D(xᵢ) ≤ τ(e)}| / N

### 3. Integration with Neural Network Training
The training objective becomes:  
min₍θ₎ (1 / |ℬₑ|) ⋅ Σ_{xᵢ ∈ ℬₑ} ℒ(f_θ(xᵢ), yᵢ)  
where ℬₑ is a mini-batch sampled uniformly from the eligible set:  
𝒮ₑ = {xᵢ ∈ 𝒟 : D(xᵢ) ≤ τ(e)}

---

## Methodology

### Phase 1: Topic Modeling & Difficulty Annotation
1.  **Feature Extraction:**
    *   *Text:* Bag-of-words or TF-IDF vectors.
    *   *Images:* Extract deep features from a pre-trained, frozen backbone (e.g., ResNet-18 penultimate layer) to create "visual word" histograms or embedding vectors.
2.  **Model Fitting:** Apply topic modeling (LDA/NMF) to the feature matrix to obtain \( P(t | x_i) \) for all \( x_i \).
3.  **Difficulty Scoring:** Compute \( D(x_i) \) for each sample (default: topic entropy).

### Phase 2: Curriculum Construction
1.  Sort the entire dataset \( \mathcal{D} \) by \( D(x_i) \) in ascending order (easiest to hardest).
2.  Define a schedule function \( \tau(e) \) (e.g., linear, root).
3.  For each epoch \( e \), create the eligible subset \( \mathcal{S}_e \).

### Phase 3: Neural Network Training
Train the target model (e.g., ResNet-18, BERT) using standard optimization (SGD, Adam), but where mini-batches are drawn from \( \mathcal{S}_e \) instead of the full \( \mathcal{D} \).

**Comparison Regimes:**
- **RS (Random Sampling):** Standard uniform shuffling.
- **Heuristic-CL:** Baseline curriculum using task-specific heuristics (e.g., sentence length for NLP, image complexity for Vision).
- **SPL (Self-Paced Learning):** Loss-based curriculum baseline.
- **TMCL (Proposed):** Curriculum based on topic-modeled difficulty \( D(x_i) \).

---

## Experimental Design

### Models & Datasets
| Domain | Dataset | Model | Task | Topic Features |
| :--- | :--- | :--- | :--- | :--- |
| **Vision** | CIFAR-10/100 | ResNet-18/34 | Classification | ResNet-18 embeddings, clustered |
| **Vision** | MNIST/F-MNIST | Simple CNN | Classification | Raw pixels (flattened) or CNN embeddings |
| **NLP** | AG News | BERT-base | Classification | BERT [CLS] embeddings or BoW |
| **NLP** | IMDb | BERT-base | Sentiment Analysis | BERT [CLS] embeddings or BoW |

### Evaluation Metrics
*   **Primary:** Test Accuracy, Macro F1-Score.
*   **Curriculum Efficacy:**
    *   Convergence Speed: \( \text{Epochs to Acc.} = \min e \text{ s.t. } \text{Acc}(e) \geq \alpha \cdot \text{Acc}_{\text{final}} \).
    *   Area Under the Training Curve (AUTC): Integral of accuracy vs. epoch curve (higher is better).
    *   Training Smoothness: \( \frac{1}{E-1}\sum_{e=1}^{E-1} |\mathcal{L}_{e+1} - \mathcal{L}_e| \) (lower is better).

### Planned Experiments
1.  **Difficulty Metric Validation:** Scatter plot and correlation analysis (\( r \)) between \( D(x_i) \) and the sample's loss after the first training epoch.
2.  **Ablation on Difficulty Metric:** Compare \( D_{\text{entropy}} \), \( D_{\text{max}} \), \( D_{\text{comp}} \) within the TMCL framework.
3.  **Curriculum Schedule Ablation:** Test linear, root (\( \gamma=0.5, 0.7 \)), and exponential (\( \beta=0.95, 0.99 \)) schedules.
4.  **Cross-Domain Benchmark:** Run the full RS, Heuristic-CL, SPL, TMCL comparison across all dataset/model pairs.
5.  **Sensitivity Analysis:** Investigate the impact of the number of topics \( T \) on TMCL performance.

---

## Expected Contributions & Significance

1.  **A Novel, Unsupervised Difficulty Metric:** Proposes and validates topic distribution entropy as a general, data-driven measure of sample complexity.
2.  **A Scalable CL Framework:** Provides a practical TMCL pipeline that requires no handcrafted rules or auxiliary models, making CL accessible for new domains.
3.  **Empirical Evidence:** A comprehensive benchmark demonstrating the conditions under which TMCL provides benefits over established training paradigms.
4.  **Theoretical Insight:** Contributes to the understanding of how the semantic structure of data interacts with neural network optimization dynamics.

---

## Future Work Directions

*   **Dynamic Topic Models:** Incorporate online/dynamic topic models (e.g., Dynamic LDA) to allow the curriculum to adapt to the model's changing understanding during training.
*   **Multi-Modal TMCL:** Extend the framework to multi-modal data (e.g., image-caption pairs) by modeling joint topic distributions across modalities.
*   **Large-Scale Language Model Training:** Investigate the application of TMCL for pre-training or fine-tuning LLMs, where curriculum learning could reduce computational cost.
*   **Continual & Lifelong Learning:** Explore TMCL for task ordering in continual learning scenarios, where "topic" spaces could represent tasks or skills.

---

## Related Work (Key Citations)

*   **Curriculum Learning:** Bengio et al., *"Curriculum Learning"* (ICML 2009); Soviany et al., *"Curriculum Learning: A Survey"* (2021).
*   **Self-Paced Learning:** Kumar et al., *"Self-Paced Learning for Latent Variable Models"* (NIPS 2010); Hacohen & Weinshall, *"On The Power of Curriculum Learning in Training Deep Networks"* (ICML 2019).
*   **Automated Curriculum:** Graves et al., *"Automated Curriculum Learning for Neural Networks"* (ICML 2017).
*   **Topic Modeling:** Blei et al., *"Latent Dirichlet Allocation"* (JMLR 2003); Miao et al., *"Neural Variational Inference for Text Processing"* (ICML 2016).
*   **Data-Centric AI:** Recent shifts towards understanding data order and quality; TMCL aligns with this paradigm.

---

## Tools & Libraries

*   **Deep Learning:** PyTorch, PyTorch Lightning, HuggingFace Transformers.
*   **Topic Modeling:** Gensim (LDA), Scikit-learn (NMF), OCTIS for neural topic models.
*   **Evaluation & Analysis:** Scikit-learn, Matplotlib, Seaborn, Weights & Biases (for experiment tracking).

---

## Author

**Reiyo**

**Research Focus:** Deep Learning, Representation Learning, Optimization, Data-Centric AI.

**Thesis Topic:** Topic-Modeled Curriculum Learning for Efficient and Robust Neural Network Training.

---
*This README serves as the living document for the research project. Theoretical formulations and experimental plans are subject to refinement based on ongoing results.*
