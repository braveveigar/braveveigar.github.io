---
layout: post
title: "Concepts I Learned Implementing an MLP From Scratch With NumPy"
date: 2026-07-15 00:00:00 +0900
last_modified_at: 2026-07-15 00:00:00 +0900
tags: [deep learning, numpy, mlp]
math: true
toc: true
lang: en
translation_key: mlp-from-scratch
permalink: /en/mlp-from-scratch/
---

Notes on the concepts I organized while implementing a fully-connected neural network for MNIST handwritten digit classification using only NumPy (writing forward/backward propagation myself, without a deep learning framework).

## Data Preprocessing

### Normalization
Since each value ranges from 0 to 255, I divided everything by 255 to normalize it to the 0~1 range.

### One-Hot Encoding
Converts labels like `[0,1,3,...]` into a form like `[[1,0,0,...],[0,1,0,...],...]`.

## Forward Propagation

### Fully-connected layer
<img src="/assets/images/mlp-fully-connected-layer.png" width="50%" class="align-center">

$$
Z = XW + b ( X : \text{input}, W : \text{weights}, b : \text{bias} )
$$

### ReLU Activation Function

$$
f(x)=
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## Loss Function

### Softmax
$$
p_i = \frac{e^{z_i}}{\sum_{j=1} e^{z_j}}
$$

### Cross-Entropy Loss
$$
L = -\frac{1}{N}\sum_{i}y_i\log(p_i)
$$

### ❗ **The Meaning of Combining Softmax + Cross-Entropy**
Softmax and Cross-Entropy are each complicated formulas on their own, but when combined, **differentiating them simplifies dramatically down to $p - y$**.

- $(p_i > y_i)$: decreases that score.
- $(p_i < y_i)$: increases that score.

In other words, **the gradient is computed exactly as the difference between the predicted probability and the true label, so the model learns in the right direction.**

---

#### Step 1. Define the Cross-Entropy Loss

$$
L = -\sum_{j=1}^{C} y_j \log p_j
$$

#### Step 2. Apply the Chain Rule

To differentiate with respect to $z_i$, every $p_j$ depends on $z_i$ (because of the softmax denominator), so we apply the chain rule with a sum over $j$.

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_i}
$$

**Step A**:

$$
\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}
$$

**Step B**: (the softmax derivative found earlier, with indices $i \leftrightarrow j$)

$$
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

#### Step 3. Substitute

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \left(-\frac{y_j}{p_j}\right) \cdot p_j(\delta_{ij} - p_i)
$$

$p_j$ cancels out in the numerator and denominator, giving

$$
= -\sum_{j=1}^{C} y_j(\delta_{ij} - p_i)
$$

#### Step 4. Split Into Two Sums

$$
= -\sum_{j=1}^{C} y_j \delta_{ij} + \sum_{j=1}^{C} y_j p_i
= -\sum_{j=1}^{C} y_j \delta_{ij} + p_i \sum_{j=1}^{C} y_j
$$

(since $p_i$ is constant with respect to $j$, it can be pulled outside the sum)

#### Step 5. Use Two Properties

$$
\sum_{j=1}^{C}y_j=1
$$

Also,

$$
\sum_{j=1}^{C}y_j\delta_{ij}=y_i
$$

(since $\delta_{ij}$ is 1 only when $j=i$, this is the sifting property of the Kronecker delta, leaving only the $y_i$ term)

Therefore,

$$
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=
-y_i+p_i
\\
&=
p_i-y_i
\end{aligned}
$$

## ❗ Backpropagation

- Chain Rule
    - If a value is computed through a series of functions, differentiating the final result with respect to each variable requires **multiplying the partial derivatives of each step in a chain**. Backpropagation uses this principle to pass gradients from the output layer back toward the input layer.
- Gradients of the fully-connected layer

### Notation

$$
x \in \mathbb{R}^{d},\quad W \in \mathbb{R}^{C\times d},\quad b \in \mathbb{R}^{C},\quad
z = Wx + b \in \mathbb{R}^{C}
$$

>#### $\partial L/\partial X$

$$
\frac{\partial z_i}{\partial x_k} = W_{ik}
\quad\Longrightarrow\quad
\begin{aligned}
\frac{\partial L}{\partial x_k}
&=
\sum_{i=1}^{C}\frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial x_k}
\\
&=
\sum_{i=1}^{C}(p_i-y_i)W_{ik}
\end{aligned}
$$

In matrix form:

$$
\frac{\partial L}{\partial X} = W^{\top}(p-y)
$$

>#### $\partial L/\partial W$

$$
\frac{\partial z_i}{\partial W_{ik}} = x_k
\quad\Longrightarrow\quad
\frac{\partial L}{\partial W_{ik}} = (p_i-y_i)\,x_k
$$

In matrix form (outer product):

$$
\frac{\partial L}{\partial W} = (p-y)\,x^{\top}
$$

>#### $\partial L/\partial b$

$$
\frac{\partial z_i}{\partial b_i} = 1
\quad\Longrightarrow\quad
\frac{\partial L}{\partial b_i} = (p_i-y_i)\cdot 1 = p_i-y_i
$$

$$
\frac{\partial L}{\partial b} = p - y
$$

- With batch size $N$

$$
\frac{\partial L}{\partial b}=\sum_{n=1}^{N}(p^{(n)}-y^{(n)})
$$

- Gradient of ReLU

$$
f'(x)=
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## Weight Initialization

### He Initialization

**Background**: If the variance of the activations changes every time they pass through a layer (exploding/vanishing), training a deep neural network becomes unstable. So the goal is to **preserve the variance**.

**Variance propagation formula**:

$$
\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

($n$ = input size, fan-in)

**Effect of ReLU**: Since it zeroes out negative values, the output variance is reduced by half.

$$
\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)
$$

**Compensation**: To preserve the variance, $\text{Var}(z)$ must be doubled.

$$
n \cdot \text{Var}(W) = 2 \quad\Longrightarrow\quad \text{Var}(W) = \frac{2}{n}
$$

**Standard deviation (used in the actual implementation)**:

$$
\sigma_W = \sqrt{\frac{2}{n}}
$$

| Initialization | Variance | Target activation |
|---|---|---|
| Xavier | $1/n$ | sigmoid, tanh |
| He | $2/n$ | ReLU |

## Training

### Parameter Updates via Gradient Descent

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

($\theta$ : trainable parameters $W, b$, $\eta$ : learning rate)

Parameters are moved a small step in the direction that decreases the loss, i.e. the **opposite direction** of the gradient, to minimize the loss.

### Mini-batch SGD

Instead of computing the gradient over the entire dataset at once, this approach approximates the gradient using only a subset of samples (a mini-batch) for each update.

- Full dataset (batch GD): the gradient is accurate but slow and memory-intensive
- One sample at a time (pure SGD): fast, but the gradient is noisy and unstable
- Mini-batch: a compromise. Sample `batch_size` items and update with the averaged gradient

$$
\frac{\partial L}{\partial \theta} \approx \frac{1}{B}\sum_{n=1}^{B} \frac{\partial L^{(n)}}{\partial \theta}
$$

($B$ : batch size)

## Evaluation

### Computing Accuracy

$$
\text{Accuracy} = \frac{\text{correct}}{N}
$$

## Retrospective
- I got confused differentiating the Cross-Entropy Loss for the case where $i \neq j$.
- I need to be careful with transposes in the chain rule during backpropagation.
