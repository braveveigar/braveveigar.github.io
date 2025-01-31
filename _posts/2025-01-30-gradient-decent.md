---
layout: post
title: What is Gradient Descent?
date: 2024-04-26 23:18 +0800
last_modified_at: 2025-01-30 11:34:25 +0800
tags: [mathematical optimization, numpy]
math: true
toc:  true
---


# Gradient Descent?

Let’s suppose you're lost somewhere in the mountains. In a situation where you don’t know exactly where you are, the best way to find your way down might be to simply follow the slope. The steeper the slope, the quicker you’ll reach the lowest point. This post explains an algorithm called gradient descent, which solves optimization problems, such as those in convolution functions.

>**Where exactly is it used?** 🤔   
In machine learning, a loss function is used to calculate the difference between the predicted values and the actual values. Gradient descent is then used to find the minimum of the loss function.

---

## The Concept of Gradient Descent

### Descent


>(a) \\\(x_{k+1} = x_{k} + t_k \Delta{x_k},k=0,1,...\\\) (where \\\(t_k>0\\\))   
(b) \\\(f(x_{k+1})<f(x_k\\\))

The method of solving an optimization problem using the conditions provided by the equation is called gradient descent. In other words, the key is to set a sequence in such a way that the function value continues to decrease as \\\(k\\\) increases.

---

### Gradient Descent

The objective function used in gradient descent is a convex function. By definition of convex function,

\\\(f(x_{k+1}) \geq f(x_k)+\nabla f(x_k)^T(x_{k+1}-x_k)\\\)

is satisfied. For the above gradient descent definition (b) to hold,

\\\(\nabla f(x_k)^T(x_{k+1}-x_k)<0\\\) and by using equation (a), this simplifies to

\\\(\nabla f(x_k)^T t_k\Delta{x_k}<0\\\)   

Since \\\(t_k\\\) is positive, we will set \\\(\Delta{x_k}\\\).

> ❗️ For a multivariable function \\\(f:R^n \rightarrow R\\\), the direction of fastest increase is the direction of \\\(\nabla f\\\).
> 

Reversing the above reasoning, if we set \\\(\Delta{x_k} = -\nabla f(x_k)\\\),we can see that \\\({x_k}\\\)moves in the direction of fastest decrease.

Now, let's implement this in Python and verify it with a graph.

## 🧑🏻‍💻 Gradient Descent with code

> Given \\\(f(x) = 2x^2+3xy+4y^2\\\), solve \\\(minimize_{x \in R^2} f(x)\\\)using gradient descent. (where \\\(x_0=(2,4), t_k=0.01, \epsilon=10^{-8}\\\))
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

![Figure 1](assets/images/gradient_descent_01.png)

By running the gradient descent 520 times, we can confirm that it converges to the solution (0,0).

---

## 🧑🏻‍💻 Plotting Graph

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

![Figure 2](assets/images/gradient_descent_02.png)

Point converges to (0,0)

---