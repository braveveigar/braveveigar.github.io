---
layout: post
title: "NumPyでMLPをゼロから実装しながら整理した概念"
date: 2026-07-15 00:00:00 +0900
last_modified_at: 2026-07-15 00:00:00 +0900
tags: [深層学習, numpy, mlp]
math: true
toc: true
lang: ja
translation_key: mlp-from-scratch
permalink: /ja/mlp-from-scratch/
---

MNIST手書き数字分類のためのfully-connectedニューラルネットワークを、ディープラーニングフレームワークを使わずNumPyのみで直接実装しながら（forward/backward propagationを自分で実装しながら）整理した概念ノートです。

## データ前処理

### 正規化
各値は0〜255の範囲であるため、全体を255で割って0〜1の範囲に正規化する。

### One-Hotエンコーディング
`[0,1,3,...]`のようなラベルを`[[1,0,0,...],[0,1,0,...],...]`のような形に変換する。

## 順伝播（Forward Propagation）

### Fully-connected layer
<img src="/assets/images/mlp-fully-connected-layer.png" width="50%" class="align-center">

$$
Z = XW + b ( X : \text{input}, W : \text{weights}, b : \text{bias} )
$$

### ReLU活性化関数

$$
f(x)=
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 損失関数

### Softmax
$$
p_i = \frac{e^{z_i}}{\sum_{j=1} e^{z_j}}
$$

### Cross-Entropy Loss
$$
L = -\frac{1}{N}\sum_{i}y_i\log(p_i)
$$

### ❗ **SoftmaxとCross-Entropyを組み合わせる意味**
SoftmaxとCross-Entropyはそれぞれ複雑な数式だが、両者を組み合わせて**微分すると $p - y$ という形に極端にシンプルになる**。

- $(p_i > y_i)$：該当するscoreを減少させる。
- $(p_i < y_i)$：該当するscoreを増加させる。

つまり、**予測確率と正解ラベルの差の分だけgradientが計算され、モデルが正しい方向へ学習される。**

---

#### Step 1. Cross-Entropy Lossの定義

$$
L = -\sum_{j=1}^{C} y_j \log p_j
$$

#### Step 2. Chain Ruleの適用

$z_i$で微分するには、（softmaxの分母のため）すべての$p_j$が$z_i$に依存しているので、$j$について総和を取ってchain ruleを適用する。

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_i}
$$

**Step A**：

$$
\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}
$$

**Step B**：（先ほど求めたsoftmaxの微分、添字$i \leftrightarrow j$を入れ替え）

$$
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

#### Step 3. 代入

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \left(-\frac{y_j}{p_j}\right) \cdot p_j(\delta_{ij} - p_i)
$$

分子・分母で$p_j$が約分されて

$$
= -\sum_{j=1}^{C} y_j(\delta_{ij} - p_i)
$$

#### Step 4. 二つの和に分離

$$
= -\sum_{j=1}^{C} y_j \delta_{ij} + \sum_{j=1}^{C} y_j p_i
= -\sum_{j=1}^{C} y_j \delta_{ij} + p_i \sum_{j=1}^{C} y_j
$$

（$p_i$は$j$に関して定数なので、総和の外に出すことができる）

#### Step 5. 二つの性質を利用

$$
\sum_{j=1}^{C}y_j=1
$$

さらに、

$$
\sum_{j=1}^{C}y_j\delta_{ij}=y_i
$$

（$\delta_{ij}$は$j=i$のときのみ1となるため、Kronecker deltaのsifting propertyにより和は$y_i$の項だけが残る）

したがって、

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

## ❗ 逆伝播（Backpropagation）

- 連鎖律（Chain Rule）
    - ある値が複数の関数を経て計算される場合、最終結果を各変数について微分するには、**各段階の偏微分を連鎖的に掛け合わせる**必要がある。逆伝播はこの原理を利用し、出力層から入力層の方向へgradientを伝達する。
- Fully-connected layerのgradient

### 記法の定義

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

行列形式：

$$
\frac{\partial L}{\partial X} = W^{\top}(p-y)
$$

>#### $\partial L/\partial W$

$$
\frac{\partial z_i}{\partial W_{ik}} = x_k
\quad\Longrightarrow\quad
\frac{\partial L}{\partial W_{ik}} = (p_i-y_i)\,x_k
$$

行列（外積、outer product）形式：

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

- バッチサイズ$N$の場合

$$
\frac{\partial L}{\partial b}=\sum_{n=1}^{N}(p^{(n)}-y^{(n)})
$$

- ReLUのgradient

$$
f'(x)=
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 重み初期化

### He Initialization

**背景**：レイヤーを通過するたびに活性値の分散が変化すると（発散/消失）、深いニューラルネットワークの学習が不安定になる。そのため**分散を維持する**ことが目標となる。

**分散伝播の式**：

$$
\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

（$n$ = input size、fan-in）

**ReLUの影響**：負の値を0にしてしまうため、出力の分散が半分に減少する。

$$
\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)
$$

**補償**：分散を維持するために、$\text{Var}(z)$を2倍にする必要がある。

$$
n \cdot \text{Var}(W) = 2 \quad\Longrightarrow\quad \text{Var}(W) = \frac{2}{n}
$$

**標準偏差（実際の実装で使用）**：

$$
\sigma_W = \sqrt{\frac{2}{n}}
$$

| 初期化 | 分散 | 対象活性化関数 |
|---|---|---|
| Xavier | $1/n$ | sigmoid, tanh |
| He | $2/n$ | ReLU |

## 学習

### 勾配降下法によるパラメータ更新

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

（$\theta$：学習パラメータ$W, b$、$\eta$：learning rate）

Lossが減少する方向、すなわちgradientの**反対方向**にパラメータを少しずつ移動させ、lossを最小化する。

### Mini-batch SGD

データセット全体で一度にgradientを計算する代わりに、一部のサンプル（mini-batch）だけでgradientを近似して更新する方式。

- データ全体（batch GD）：gradientは正確だが、遅くメモリ負担が大きい
- サンプル1つずつ（純粋なSGD）：速いが、gradientが不安定（noisy）
- Mini-batch：折衷案。`batch_size`だけ抽出し、平均gradientで更新する

$$
\frac{\partial L}{\partial \theta} \approx \frac{1}{B}\sum_{n=1}^{B} \frac{\partial L^{(n)}}{\partial \theta}
$$

（$B$：batch size）

## 評価

### Accuracyの計算

$$
\text{Accuracy} = \frac{\text{correct}}{N}
$$

## 振り返り
- Cross Entropy Lossの偏微分でi=jでない場合の微分に混乱した。
- 逆伝播のchain ruleにおいて転置に注意が必要。
