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

## Evaluation metrics
Here's the translation of the provided Vietnamese text into English:

---

In recommender systems, evaluation metrics are commonly used to measure the performance of the recommendation model. Below are some popular metrics along with examples illustrating how to calculate each one:

1. **Precision**  
Precision measures the proportion of recommended items that are relevant to the user.

\[ \text{Precision} = \frac{|\{ \text{Relevant Items} \} \cap \{ \text{Recommended Items} \}|}{|\{ \text{Recommended Items} \}|} \]

**Example:**  
- Relevant items: \{A, B, D\}  
- Recommended items: \{A, B, C\}  

\[ \text{Precision} = \frac{|\{A, B, D\} \cap \{A, B, C\}|}{|\{A, B, C\}|} = \frac{2}{3} \approx 0.67 \]

2. **Recall**  
Recall measures the proportion of relevant items that are recommended out of all relevant items.

\[ \text{Recall} = \frac{|\{ \text{Relevant Items} \} \cap \{ \text{Recommended Items} \}|}{|\{ \text{Relevant Items} \}|} \]

**Example:**  
- Relevant items: \{A, B, D\}  
- Recommended items: \{A, B, C\}  

\[ \text{Recall} = \frac{|\{A, B, D\} \cap \{A, B, C\}|}{|\{A, B, D\}|} = \frac{2}{3} \approx 0.67 \]

3. **F1-Score**  
F1-Score is the harmonic mean of Precision and Recall.

\[ \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \]

**Example:**  
- Precision: 0.67  
- Recall: 0.67  

\[ \text{F1-Score} = 2 \cdot \frac{0.67 \cdot 0.67}{0.67 + 0.67} = 0.67 \]

4. **Mean Average Precision (MAP)**  
MAP measures the average precision across multiple cut-off thresholds.

\[ \text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{Average Precision}(q) \]

**Example:**  
- User 1: Precision = 0.67  
- User 2: Precision = 0.50  

\[ \text{MAP} = \frac{1}{2} (0.67 + 0.50) = 0.585 \]

5. **Normalized Discounted Cumulative Gain (NDCG)**  
NDCG evaluates the quality of recommendations based on the ranks of the relevant items.

\[ \text{NDCG} = \frac{\text{DCG}}{\text{IDCG}} \]

Where:

\[ \text{DCG} = \sum_{i=1}^{p} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)} \]

**Example:**  
- Ranks: \{3, 2, 3\}  

\[ \text{DCG} = \frac{2^3 - 1}{\log_2(1 + 1)} + \frac{2^2 - 1}{\log_2(2 + 1)} + \frac{2^3 - 1}{\log_2(3 + 1)} = 7 + 1.89 + 1.5 \approx 10.39 \]

6. **Hit Rate**  
Hit Rate measures the proportion of recommendation sessions where at least one relevant item is recommended.

\[ \text{Hit Rate} = \frac{|\{ \text{Users} \} \cap \{ \text{Users with at least one relevant item recommended} \}|}{|\{ \text{Users} \}|} \]

**Example:**  
- 5 users, 4 users have at least 1 relevant item recommended.

\[ \text{Hit Rate} = \frac{4}{5} = 0.8 \]

7. **Coverage**  
Coverage measures the proportion of items in the dataset that can be recommended.

\[ \text{Coverage} = \frac{|\{ \text{Items recommended} \}|}{|\{ \text{Total items} \}|} \]

**Example:**  
- 100 total items, 80 items recommended at least once.

\[ \text{Coverage} = \frac{80}{100} = 0.8 \]

8. **Diversity**  
Diversity measures the variety of items recommended to the user.

\[ \text{Diversity} = 1 - \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} \text{Similarity}(i,j) \]

**Example:**  
- \( n = 3 \), Similarity scores: 0.2, 0.3, 0.4

\[ \text{Diversity} = 1 - \frac{1}{3(3-1)} (0.2 + 0.3 + 0.4) = 1 - \frac{0.9}{6} = 1 - 0.15 = 0.85 \]

9. **Mean Reciprocal Rank (MRR)**  
MRR measures the average position of the first relevant item in the recommendation list.

\[ \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \]

**Example:**  
- User 1: rank = 1  
- User 2: rank = 2  

\[ \text{MRR} = \frac{1}{2} \left( \frac{1}{1} + \frac{1}{2} \right) = \frac{1}{2} \left( 1 + 0.5 \right) = 0.75 \]

10. **Root Mean Square Error (RMSE)**  
RMSE measures the difference between actual values and predicted values.

\[ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2} \]

**Example:**  
- Actual values: \{3, 4, 5\}  
- Predicted values: \{3.5, 4, 4.5\}  

\[ \text{RMSE} = \sqrt{\frac{1}{3} ((3-3.5)^2 + (4-4)^2 + (5-4.5)^2)} = \sqrt{0.167} \approx 0.41 \]

11. **Mean Absolute Error (MAE)**  
MAE measures the average deviation between actual values and predicted values.

\[ \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y_i}| \]

**Example:**  
- Actual values: \{3, 4, 5\}  
- Predicted values: \{3.5, 4, 4.5\}  

\[ \text{MAE} = \frac{1}{3} (|3-3.5| + |4-4| + |5-4.5|) = 0.33 \]

These metrics provide different perspectives on the performance of recommender systems and are often used together to get a comprehensive view of the recommendation quality.


- [Understanding Recommender Systems](https://lnkd.in/gqUhE2ny)
