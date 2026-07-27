---
layout: post
title: "什么是梯度下降？"
date: 2024-04-26 23:18 +0800
last_modified_at: 2025-01-30 11:34:25 +0800
tags: [数学优化, numpy]
math: true
toc: true
lang: zh
translation_key: gradient-descent
permalink: /zh/gradient-descent/
---

假设你迷失在某座山中。在不清楚自己确切位置的情况下，找到下山之路最好的方法或许就是顺着坡度往下走。坡度越陡，你就能越快到达最低点。这篇文章将介绍一种用于求解优化问题（例如凸函数中的优化问题）的算法——梯度下降法（gradient descent）。

>**它究竟用在哪里？** 🤔   
在机器学习中，损失函数（loss function）用于计算预测值与真实值之间的差异。梯度下降法就是用来求损失函数最小值的方法。

---

## 梯度下降的概念

### 下降


>(a) \\\(x_{k+1} = x_{k} + t_k \Delta{x_k},k=0,1,...\\\)（其中 \\\(t_k>0\\\)）   
(b) \\\(f(x_{k+1})<f(x_k\\\))

利用该方程所给出的条件来求解优化问题的方法称为梯度下降法。换句话说，关键在于设定一个数列，使得函数值随着 \\\(k\\\) 的增加而持续减小。

---

### 梯度下降

梯度下降法所使用的目标函数是一个凸函数（convex function）。根据凸函数的定义，

\\\(f(x_{k+1}) \geq f(x_k)+\nabla f(x_k)^T(x_{k+1}-x_k)\\\)

成立。要使上述梯度下降的定义 (b) 成立，

\\\(\nabla f(x_k)^T(x_{k+1}-x_k)<0\\\)，并利用方程 (a)，可以简化为

\\\(\nabla f(x_k)^T t_k\Delta{x_k}<0\\\)   

由于 \\\(t_k\\\) 为正数，我们将设定 \\\(\Delta{x_k}\\\)。

> ❗️ 对于多变量函数 \\\(f:R^n \rightarrow R\\\)，增长最快的方向是 \\\(\nabla f\\\) 的方向。
> 

反过来推导，如果我们设定 \\\(\Delta{x_k} = -\nabla f(x_k)\\\)，可以看出 \\\({x_k}\\\) 正朝着下降最快的方向移动。

现在，让我们用 Python 实现它，并通过图形加以验证。

## 🧑🏻‍💻 用代码实现梯度下降

> 给定 \\\(f(x) = 2x^2+3xy+4y^2\\\)，使用梯度下降法求解 \\\(minimize_{x \in R^2} f(x)\\\)。（其中 \\\(x_0=(2,4), t_k=0.01, \epsilon=10^{-8}\\\)）
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

将梯度下降运行 520 次后，我们可以确认它收敛到了解 (0,0)。

---

## 🧑🏻‍💻 绘制图形

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

收敛到 (0,0) 的点

---
