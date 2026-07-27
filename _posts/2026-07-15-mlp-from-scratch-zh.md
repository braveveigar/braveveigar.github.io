---
layout: post
title: "用 NumPy 从零实现 MLP 时整理的概念"
date: 2026-07-15 00:00:00 +0900
last_modified_at: 2026-07-15 00:00:00 +0900
tags: [深度学习, numpy, mlp]
math: true
toc: true
lang: zh
translation_key: mlp-from-scratch
permalink: /zh/mlp-from-scratch/
---

这是在不使用深度学习框架、仅用 NumPy 亲手实现用于 MNIST 手写数字分类的全连接神经网络（自己编写前向/反向传播）过程中整理的概念笔记。

## 数据预处理

### 归一化
每个数值都在 0~255 之间，因此将其整体除以 255，归一化到 0~1 的范围。

### 独热编码（One-Hot Encoding）
将 `[0,1,3,...]` 这样的标签转换为 `[[1,0,0,...],[0,1,0,...],...]` 的形式。

## 前向传播（Forward Propagation）

### 全连接层（Fully-connected layer）
<img src="/assets/images/mlp-fully-connected-layer.png" width="50%" class="align-center">

$$
Z = XW + b ( X : \text{input}, W : \text{weights}, b : \text{bias} )
$$

### ReLU 激活函数

$$
f(x)=
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 损失函数

### Softmax
$$
p_i = \frac{e^{z_i}}{\sum_{j=1} e^{z_j}}
$$

### 交叉熵损失（Cross-Entropy Loss）
$$
L = -\frac{1}{N}\sum_{i}y_i\log(p_i)
$$

### ❗ **Softmax 与 Cross-Entropy 结合的意义**
Softmax 函数和 Cross-Entropy 函数各自都是较为复杂的公式，但将两者结合后，**求导时会极度简化为 $p - y$**。

- $(p_i > y_i)$：降低该 score。
- $(p_i < y_i)$：提高该 score。

也就是说，**梯度恰好等于预测概率与真实标签之间的差值，从而使模型朝正确的方向学习。**

---

#### 步骤 1：定义交叉熵损失

$$
L = -\sum_{j=1}^{C} y_j \log p_j
$$

#### 步骤 2：应用链式法则（Chain Rule）

由于所有的 $p_j$（由于 softmax 的分母）都依赖于 $z_i$，因此要对 $z_i$ 求导，需要对 $j$ 求和后应用链式法则。

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_i}
$$

**步骤 A**：

$$
\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}
$$

**步骤 B**：（前面求得的 softmax 导数，交换索引 $i \leftrightarrow j$）

$$
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

#### 步骤 3：代入

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \left(-\frac{y_j}{p_j}\right) \cdot p_j(\delta_{ij} - p_i)
$$

$p_j$ 在分子分母中相互约去，得到

$$
= -\sum_{j=1}^{C} y_j(\delta_{ij} - p_i)
$$

#### 步骤 4：拆分为两个求和

$$
= -\sum_{j=1}^{C} y_j \delta_{ij} + \sum_{j=1}^{C} y_j p_i
= -\sum_{j=1}^{C} y_j \delta_{ij} + p_i \sum_{j=1}^{C} y_j
$$

（因为 $p_i$ 相对于 $j$ 是常数，所以可以提到求和符号外面）

#### 步骤 5：利用两个性质

$$
\sum_{j=1}^{C}y_j=1
$$

此外，

$$
\sum_{j=1}^{C}y_j\delta_{ij}=y_i
$$

（$\delta_{ij}$ 仅在 $j=i$ 时为 1，这是 Kronecker delta 的筛选性质，因此求和后只剩下 $y_i$ 一项）

因此，

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

## ❗ 反向传播（Backpropagation）

- 链式法则（Chain Rule）
    - 如果某个值是经过多个函数计算得到的，那么对最终结果关于各个变量求导时，需要**将每一步的偏导数连乘**。反向传播正是利用这一原理，将梯度从输出层向输入层方向传递。
- 全连接层的梯度

### 符号定义

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

矩阵形式：

$$
\frac{\partial L}{\partial X} = W^{\top}(p-y)
$$

>#### $\partial L/\partial W$

$$
\frac{\partial z_i}{\partial W_{ik}} = x_k
\quad\Longrightarrow\quad
\frac{\partial L}{\partial W_{ik}} = (p_i-y_i)\,x_k
$$

矩阵（外积，outer product）形式：

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

- 批次大小为 $N$ 时

$$
\frac{\partial L}{\partial b}=\sum_{n=1}^{N}(p^{(n)}-y^{(n)})
$$

- ReLU 的梯度

$$
f'(x)=
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 权重初始化

### He 初始化（He Initialization）

**背景**：如果激活值的方差在每经过一层后都发生变化（爆炸/消失），深层神经网络的训练就会变得不稳定。因此目标是**保持方差不变**。

**方差传播公式**：

$$
\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

（$n$ = 输入维度，fan-in）

**ReLU 的影响**：由于它将负值置为 0，输出的方差会减半。

$$
\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)
$$

**补偿**：为了保持方差不变，需要将 $\text{Var}(z)$ 放大 2 倍。

$$
n \cdot \text{Var}(W) = 2 \quad\Longrightarrow\quad \text{Var}(W) = \frac{2}{n}
$$

**标准差（实际实现中使用）**：

$$
\sigma_W = \sqrt{\frac{2}{n}}
$$

| 初始化 | 方差 | 目标激活函数 |
|---|---|---|
| Xavier | $1/n$ | sigmoid, tanh |
| He | $2/n$ | ReLU |

## 训练

### 通过梯度下降更新参数

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

（$\theta$：待训练参数 $W, b$，$\eta$：学习率）

沿着损失减小的方向，即梯度的**反方向**，一点点地移动参数，以最小化损失。

### Mini-batch SGD

不是用整个数据集一次性计算梯度，而是仅使用一部分样本（mini-batch）来近似梯度并进行更新的方法。

- 全量数据（batch GD）：梯度准确，但速度慢、内存开销大
- 每次一个样本（纯 SGD）：速度快，但梯度不稳定（noisy）
- Mini-batch：折中方案。抽取 `batch_size` 个样本，用平均梯度进行更新

$$
\frac{\partial L}{\partial \theta} \approx \frac{1}{B}\sum_{n=1}^{B} \frac{\partial L^{(n)}}{\partial \theta}
$$

（$B$：批次大小）

## 评估

### 计算准确率（Accuracy）

$$
\text{Accuracy} = \frac{\text{correct}}{N}
$$

## 回顾
- 在对 Cross Entropy Loss 求偏导时，$i \neq j$ 的情况让我感到困惑。
- 在反向传播的链式法则中，要注意转置（transpose）。
