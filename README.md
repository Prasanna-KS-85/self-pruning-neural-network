# Self-Pruning Neural Network

## Overview

Deep neural networks are often over-parameterized, making them computationally expensive and difficult to deploy in resource-constrained environments. A common approach is post-training pruning, where unnecessary weights are removed after training.

This project implements a **self-pruning neural network**, where the model learns to identify and remove unimportant connections during training itself using learnable gating mechanisms.

---

## Key Idea

Each weight in the network is associated with a learnable gate parameter. The effective weight is computed as:

W' = W × g

Where:

* W is the original weight
* g is a gate value in the range [0, 1]

If g approaches zero, the corresponding weight is effectively pruned.

---

## Methodology

### Prunable Linear Layer

A custom linear layer is implemented with:

* Standard weight and bias
* Learnable `gate_scores`

Gates are computed using a sigmoid function:

g = sigmoid(temperature × gate_scores)

A temperature factor is used to sharpen the gating behavior, enabling stronger separation between active and pruned connections.

---

### Loss Function

The total loss is defined as:

Total Loss = CrossEntropy Loss + λ × Sparsity Loss

Where:

* Sparsity Loss is the L1 norm of gate values
* A normalized L1 penalty is used to stabilize training

This encourages the model to minimize the number of active connections.

---

### Sparsity Measurement

Sparsity is defined as:

Sparsity (%) = (Number of gates below threshold / Total gates) × 100

* Threshold used: 0.1 (for interpretability)
* Lower thresholds (e.g., 1e-2) produce similar trends

---

## Model Architecture

A Multi-Layer Perceptron (MLP) is used:

* Input: 3072 (flattened CIFAR-10 image)
* Hidden Layer 1: 512 units (PrunableLinear)
* Hidden Layer 2: 256 units (PrunableLinear)
* Output Layer: 10 classes

The MLP architecture was chosen to isolate and analyze the pruning mechanism without convolutional complexity.

---

## Experimental Setup

* Dataset: CIFAR-10
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 128
* Epochs: 20

Experiments were conducted with different λ values to analyze the trade-off between sparsity and accuracy.

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 1.0        | 54.74             | 53.29        |
| 5.0        | 56.44             | 88.78        |
| 10.0       | 55.92             | 95.39        |

---

## Analysis

* Increasing λ leads to higher sparsity, as expected.
* Moderate sparsity (λ = 5.0) achieves the best performance.
* High sparsity (λ = 10.0) significantly compresses the model while maintaining reasonable accuracy.

This demonstrates a clear trade-off between model efficiency and representational capacity.

---

## Gate Behavior

The distribution of gate values shows:

* A large concentration near zero (pruned connections)
* A smaller cluster away from zero (important connections)

This indicates successful separation between useful and redundant weights.

---

## Notebook Structure

All implementation details are contained in a single notebook:

* Data loading and preprocessing
* Custom PrunableLinear layer
* Model definition
* Training loop with sparsity loss
* Evaluation and sparsity computation
* Visualization of gate distributions

---

## How to Run

1. Clone the repository:

   ```
   git clone <your-repo-link>
   cd <repo-name>
   ```

2. Install dependencies:

   ```
   pip install torch torchvision matplotlib
   ```

3. Open the notebook:

   ```
   jupyter notebook self_pruning_network.ipynb
   ```

4. Run all cells sequentially

---

## Key Insights

* The network dynamically learns which connections are important
* L1 regularization on gates promotes sparsity
* Temperature scaling improves pruning effectiveness
* Moderate pruning can improve generalization

---

## Conclusion

This project demonstrates that neural networks can learn to prune themselves during training, resulting in more efficient and compact models. The approach provides a practical balance between performance and computational efficiency.
