# MULTIMODAL-RECSYS-KNOWLEDGE
# Multimodal Recommender Systems

## Table of Contents

1. [Introduction](#introduction)
2. [Feature Interaction](#feature-interaction)
   - [Combined Attention](#combined-attention)
   - [Fine-grained Attention](#fine-grained-attention)
   - [Knowledge Graph](#knowledge-graph)
   - [Coarse-grained Attention](#coarse-grained-attention)
   - [User-item Graph](#user-item-graph)
   - [Item-item Graph](#item-item-graph)
   - [Filtration](#filtration)
   - [Other Fusion](#other-fusion)
3. [Objective Functions](#objective-functions)
   - [Mean Square Error (MSE)](#mean-square-error-mse)
   - [Bayesian Personalized Ranking (BPR) Loss](#bayesian-personalized-ranking-bpr-loss)
   - [Cross Entropy Loss](#cross-entropy-loss)
   - [Weighted Approximate-Rank Pairwise (WARP) Loss](#weighted-approximate-rank-pairwise-warp-loss)
   - [Contrastive Loss (InfoNCE Loss)](#contrastive-loss-infonce-loss)
4. [Datasets](#datasets)
5. [References](#references)

## Introduction

Multimodal recommender systems utilize various types of data (e.g., text, images, and graphs) to provide personalized recommendations. This document covers different aspects of these systems, including feature interaction techniques, objective functions, and commonly used datasets.

## Feature Interaction

### Combined Attention

**Description:** Combined attention uses multiple attention mechanisms to capture interactions between different features.

**Pros:**
- Handles complex interactions.
- Can be fine-tuned for specific tasks.

**Cons:**
- Computationally expensive.
- Requires large datasets for effective training.

### Fine-grained Attention

**Description:** Fine-grained attention focuses on detailed interactions between features.

**Pros:**
- Provides high precision.
- Suitable for tasks requiring detailed analysis.

**Cons:**
- Computationally intensive.
- Can overfit on small datasets.

### Knowledge Graph

**Description:** Utilizes knowledge graphs to enhance recommendations by incorporating external knowledge.

**Pros:**
- Rich contextual information.
- Improves recommendation diversity.

**Cons:**
- Complex to implement.
- Requires maintenance of the knowledge graph.

### Coarse-grained Attention

**Description:** Coarse-grained attention captures broad interactions between features.

**Pros:**
- Less computationally intensive than fine-grained attention.
- Easier to train.

**Cons:**
- May miss subtle interactions.
- Lower precision.

### User-item Graph

**Description:** Represents users and items as nodes in a graph, with edges representing interactions.

**Pros:**
- Captures relational data.
- Scalable to large datasets.

**Cons:**
- Requires graph construction.
- Sensitive to graph sparsity.

### Item-item Graph

**Description:** Represents items as nodes and uses edges to represent similarities between items.

**Pros:**
- Effective for item similarity.
- Simple to implement.

**Cons:**
- Limited to item-item interactions.
- Can suffer from cold-start problems.

### Filtration

**Description:** Filters features to retain only the most relevant ones for recommendations.

**Pros:**
- Reduces dimensionality.
- Improves computational efficiency.

**Cons:**
- Risk of losing important information.
- Requires careful tuning.

### Other Fusion

**Description:** Combines multiple fusion techniques to enhance feature interactions.

**Pros:**
- Flexible and adaptable.
- Can leverage strengths of multiple methods.

**Cons:**
- Increased complexity.
- Harder to interpret results.

## Objective Functions

### Mean Square Error (MSE)

**Formula:**

\[ \text{MSE} = \frac{1}{m} \sum_{i=1}^m (h(\mathbf{x}_i) - y_i)^2 \]

**Example:**

- Actual values: [4, 3, 5]
- Predicted values: [3.5, 2.5, 5]

\[ \text{MSE} = \frac{1}{3} ((3.5-4)^2 + (2.5-3)^2 + (5-5)^2) = 0.167 \]

### Bayesian Personalized Ranking (BPR) Loss

**Formula:**

\[ L_{\text{bpr}} = -\sum_{(u,i,i') \in R} \ln \sigma(\mathbf{u}^\top \mathbf{i} - \mathbf{u}^\top \mathbf{i'}) \]

**Example:**

- \( \mathbf{u}^\top \mathbf{i} = 2 \)
- \( \mathbf{u}^\top \mathbf{i'} = 1 \)

\[ L_{\text{bpr}} = -\ln \sigma(2 - 1) \approx 0.313 \]

### Cross Entropy Loss

**Formula:**

\[ L_{\text{ce}} = -\sum_{(u,i) \in R} y_{ui} \cdot \log(\mathbf{u}^\top \mathbf{i}) + (1 - y_{ui}) \cdot \log(1 - \mathbf{u}^\top \mathbf{i}) \]

**Example:**

- Predicted value: \( \mathbf{u}^\top \mathbf{i} = 0.8 \)
- Actual value: \( y_{ui} = 1 \)

\[ L_{\text{ce}} \approx 0.223 \]

### Weighted Approximate-Rank Pairwise (WARP) Loss

**Explanation:**

WARP loss penalizes a positive item at a lower rank much more heavily than one at the top.

### Contrastive Loss (InfoNCE Loss)

**Formula:**

\[ L_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{u}^\top \mathbf{i} / \tau)}{\sum_{i' \in N_u} \exp(\mathbf{u}^\top \mathbf{i'} / \tau)} \]

**Example:**

- \( \mathbf{u}^\top \mathbf{i} = 0.8 \)
- \( \mathbf{u}^\top \mathbf{i'} = 0.3 \)
- \( \tau = 0.1 \)

\[ L_{\text{InfoNCE}} \approx 0.0068 \]

## Datasets

| Dataset | # Users | # Items | # Interactions | Sparsity |
|---------|---------|---------|----------------|----------|
| Baby    | 19,445  | 7,050   | 35,598         | 99.8827% |
| Sports  | 18,357  | 61,668  | 21,874         | 99.9547% |
| FoodRec | 192,403 | 63,001  | 160,792        | 99.8774% |
| Elec    | 296,337 | 1,654,456 | 1,689,188     | 99.9861% |

## References

- [Understanding Recommender Systems](https://lnkd.in/gqUhE2ny)
