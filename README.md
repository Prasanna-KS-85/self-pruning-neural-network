# Self-Pruning Neural Network

## Overview

Modern deep neural networks are often over-parameterized, making them computationally expensive and difficult to deploy on resource-constrained systems. A common solution is post-training pruning, where unimportant weights are removed after training.

This project implements a **self-pruning neural network**, where the model learns to identify and remove unnecessary connections during training itself. The network dynamically adapts its structure by learning which weights are important and which can be pruned.

---

## Key Idea

Each weight in the network is associated with a learnable **gate parameter**. The effective weight is computed as:

W' = W × g

Where:
- W is the original weight
- g is the gate value in the range [0, 1]

If g approaches 0, the corresponding weight is effectively removed from the network.

---

## Methodology

### 1. Prunable Linear Layer

A custom `PrunableLinear` layer is implemented with:
- Standard weight and bias parameters
- A learnable `gate_scores` tensor

Gates are computed using a sigmoid function:

g = sigmoid(temperature × gate_scores)

Temperature scaling is introduced to sharpen the gating behavior and enable more effective pruning.

---

### 2. Loss Function

The total loss combines classification performance with sparsity regularization:

Total Loss = CrossEntropy Loss + λ × Sparsity Loss

Where:
- Sparsity Loss is the L1 norm of gate values
- A normalized L1 penalty is used for stability across layers

This encourages the network to minimize the number of active connections.

---

### 3. Sparsity Measurement

Sparsity is defined as the percentage of gates below a threshold:

Sparsity (%) = (Number of gates < threshold / Total gates) × 100

In this implementation:
- Threshold = 0.1 (for interpretability)
- Lower thresholds (e.g., 1e-2) yield consistent trends

---

## Model Architecture

A simple Multi-Layer Perceptron (MLP) is used:

- Input: 3072 (flattened CIFAR-10 image)
- Hidden Layer 1: 512 units (PrunableLinear)
- Hidden Layer 2: 256 units (PrunableLinear)
- Output Layer: 10 classes

An MLP was intentionally chosen to isolate and clearly observe the pruning mechanism without convolutional complexity.

---

## Experimental Setup

- Dataset: CIFAR-10
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 128
- Epochs: 20

Experiments were conducted with different values of λ to analyze the trade-off between sparsity and accuracy.

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|------------|------------------|--------------|
| 1.0        | 54.74            | 53.29        |
| 5.0        | 56.44            | 88.78        |
| 10.0       | 55.92            | 95.39        |

---

## Analysis

- Increasing λ leads to higher sparsity, as expected.
- Moderate sparsity (λ = 5.0) achieves the best performance, suggesting that removing redundant connections improves generalization.
- High sparsity (λ = 10.0) significantly compresses the model while maintaining reasonable accuracy.

This demonstrates a clear trade-off between model efficiency and representational capacity.

---

## Gate Distribution

The distribution of gate values exhibits a bi-modal pattern:
- A large concentration near 0 (pruned connections)
- A smaller cluster away from 0 (important connections)

This indicates successful separation between useful and redundant weights.

---

## Key Insights

- The network effectively performs **dynamic model compression during training**
- L1 regularization on gates promotes sparsity
- Temperature scaling improves convergence toward sparse solutions
- Moderate pruning can improve generalization

---

## Extensions

The learned soft gates can be converted into a hard-pruned model by applying a threshold:

mask = (gates > threshold)

This enables deployment of a smaller and more efficient network.

---

## How to Run

1. Clone the repository:
