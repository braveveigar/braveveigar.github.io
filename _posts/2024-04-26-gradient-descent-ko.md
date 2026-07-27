---
layout: post
title: "경사하강법이란?"
date: 2024-04-26 23:18 +0800
last_modified_at: 2025-01-30 11:34:25 +0800
tags: [수학적 최적화, numpy]
math: true
toc: true
lang: ko
translation_key: gradient-descent
permalink: /gradient-descent/
---

산속에서 길을 잃었다고 가정해 봅시다. 정확히 어디에 있는지 모르는 상황에서, 산을 내려가는 가장 좋은 방법은 그저 경사를 따라가는 것일지도 모릅니다. 경사가 가파를수록 가장 낮은 지점에 더 빨리 도달하게 됩니다. 이 글에서는 볼록함수와 같은 최적화 문제를 푸는 경사하강법(gradient descent)이라는 알고리즘을 설명합니다.

>**정확히 어디에 쓰이나요?** 🤔   
머신러닝에서는 손실 함수(loss function)를 사용해 예측값과 실제값의 차이를 계산합니다. 이때 손실 함수의 최솟값을 찾기 위해 경사하강법이 사용됩니다.

---

## 경사하강법의 개념

### 하강


>(a) \\\(x_{k+1} = x_{k} + t_k \Delta{x_k},k=0,1,...\\\) (단, \\\(t_k>0\\\))   
(b) \\\(f(x_{k+1})<f(x_k\\\))

위 식이 제공하는 조건을 이용해 최적화 문제를 푸는 방법을 경사하강법이라고 합니다. 다시 말해, 핵심은 \\\(k\\\)가 증가함에 따라 함수값이 계속 감소하도록 수열을 설정하는 것입니다.

---

### 경사하강법

경사하강법에서 사용되는 목적함수는 볼록함수(convex function)입니다. 볼록함수의 정의에 의해,

\\\(f(x_{k+1}) \geq f(x_k)+\nabla f(x_k)^T(x_{k+1}-x_k)\\\)

가 성립합니다. 위의 경사하강법 정의 (b)가 성립하려면,

\\\(\nabla f(x_k)^T(x_{k+1}-x_k)<0\\\)이어야 하며, 식 (a)를 이용하면 다음과 같이 정리됩니다.

\\\(\nabla f(x_k)^T t_k\Delta{x_k}<0\\\)   

\\\(t_k\\\)는 양수이므로, \\\(\Delta{x_k}\\\)를 다음과 같이 설정하겠습니다.

> ❗️ 다변수 함수 \\\(f:R^n \rightarrow R\\\)에서, 가장 빠르게 증가하는 방향은 \\\(\nabla f\\\) 방향입니다.
> 

위의 논리를 반대로 적용해서 \\\(\Delta{x_k} = -\nabla f(x_k)\\\)로 설정하면, \\\({x_k}\\\)가 가장 빠르게 감소하는 방향으로 움직인다는 것을 알 수 있습니다.

이제 이를 파이썬으로 구현하고 그래프로 확인해 보겠습니다.

## 🧑🏻‍💻 코드로 구현하는 경사하강법

> \\\(f(x) = 2x^2+3xy+4y^2\\\)가 주어졌을 때, 경사하강법을 이용해 \\\(minimize_{x \in R^2} f(x)\\\)를 풀어보세요. (단, \\\(x_0=(2,4), t_k=0.01, \epsilon=10^{-8}\\\))
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

경사하강법을 520번 반복 실행한 결과, 해 (0,0)으로 수렴하는 것을 확인할 수 있습니다.

---

## 🧑🏻‍💻 그래프 그리기

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

(0,0)으로 수렴하는 지점

---
