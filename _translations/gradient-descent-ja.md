---
layout: post
title: "勾配降下法とは？"
date: 2024-04-26 23:18 +0800
last_modified_at: 2025-01-30 11:34:25 +0800
tags: [数理最適化, numpy]
math: true
toc: true
lang: ja
translation_key: gradient-descent
permalink: /ja/gradient-descent/
---

山の中で道に迷ってしまったとしましょう。自分が正確にどこにいるのか分からない状況では、下山する最良の方法は、ただ傾斜に従って進むことかもしれません。傾斜が急であればあるほど、より早く最も低い地点にたどり着けます。この記事では、凸関数のような最適化問題を解くアルゴリズムである勾配降下法（gradient descent）について説明します。

>**具体的にどこで使われるのでしょうか？** 🤔   
機械学習では、損失関数（loss function）を用いて予測値と実際の値との差を計算します。そして、損失関数の最小値を求めるために勾配降下法が使われます。

---

## 勾配降下法の概念

### 降下


>(a) \\\(x_{k+1} = x_{k} + t_k \Delta{x_k},k=0,1,...\\\)（ただし \\\(t_k>0\\\)）   
(b) \\\(f(x_{k+1})<f(x_k\\\))

この式が与える条件を用いて最適化問題を解く方法を勾配降下法と呼びます。言い換えると、重要なのは \\\(k\\\) が増加するにつれて関数値が減少し続けるように数列を設定することです。

---

### 勾配降下法

勾配降下法で用いられる目的関数は凸関数（convex function）です。凸関数の定義により、

\\\(f(x_{k+1}) \geq f(x_k)+\nabla f(x_k)^T(x_{k+1}-x_k)\\\)

が成り立ちます。上記の勾配降下法の定義 (b) が成り立つためには、

\\\(\nabla f(x_k)^T(x_{k+1}-x_k)<0\\\) であり、式 (a) を用いると、これは次のように簡略化されます。

\\\(\nabla f(x_k)^T t_k\Delta{x_k}<0\\\)   

\\\(t_k\\\) は正であるため、\\\(\Delta{x_k}\\\) を次のように設定します。

> ❗️ 多変数関数 \\\(f:R^n \rightarrow R\\\) において、最も急に増加する方向は \\\(\nabla f\\\) の方向です。
> 

上記の論理を逆にたどると、\\\(\Delta{x_k} = -\nabla f(x_k)\\\) と設定すれば、\\\({x_k}\\\) が最も急に減少する方向へ動くことが分かります。

それでは、これを Python で実装し、グラフで確認してみましょう。

## 🧑🏻‍💻 コードで実装する勾配降下法

> \\\(f(x) = 2x^2+3xy+4y^2\\\) が与えられたとき、勾配降下法を用いて \\\(minimize_{x \in R^2} f(x)\\\) を解いてください。（ただし \\\(x_0=(2,4), t_k=0.01, \epsilon=10^{-8}\\\)）
> 

```python
# objective function
def f(x,y):
    return 2*x**2 + 3*x*y + 4*y**2

# The partial derivative of the objective function with respect to x
def fx(x,y):
    return 4*x+3*y

# The partial derivative of the objective function with respect to y
def fy(x,y):
    return 3*x+8*y

# A list to store the coordinates of x_k, y_k
xlist,ylist=[2],[4]

# configuration value
x0,y0 = 2,4
t=0.01
eps=10**(-8)

iter=0
xk,yk=x0,y0

while True:
    tk=t

    # xkp1 = xk - tk * delta(xk)
    xkp1=xk-tk*fx(xk,yk)
    ykp1=yk-tk*fy(xk,yk)

    xlist.append(xkp1)
    ylist.append(ykp1)

    if np.linalg.norm(np.array((xkp1-xk,ykp1-yk)))<eps:
        print(f'iterated {iter} times')
        print(f'GD converges to {round(xkp1,1),round(ykp1,1)}')
        break

    iter = iter + 1
    xk,yk = xkp1, ykp1

```

![Figure 1](/assets/images/gradient_descent_01.png)

勾配降下法を520回実行した結果、解 (0,0) に収束することが確認できます。

---

## 🧑🏻‍💻 グラフの描画

```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import animation

# list of z coordinates from x,y
zlist=[]
for i in range(len(xlist)):
	zlist.append(f(xlist[i],ylist[i]))

# figuring graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-30)

# generating x, y
x = np.linspace(-4, 4, 10)
y = np.linspace(-4, 4, 10)
x, y = np.meshgrid(x, y)

# calculating z
z = f(x,y)

# plotting 3D
ax.plot_surface(x,y,z,color='#d070fb', alpha = 0.6)
ax.plot(xlist,ylist,zlist,lw=3, color='black')
ax.text(xlist[0],ylist[0],zlist[0],'(x0,y0,f(x0,y0))')
ax.text(xlist[-1],ylist[-1],zlist[-1],'Solution')

# Labeling x, y, z
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()

```

![Figure 2](/assets/images/gradient_descent_02.png)

(0,0) に収束する点

---
