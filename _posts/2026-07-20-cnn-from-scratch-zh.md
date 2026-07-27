---
layout: post
title: "用 NumPy 从零实现 CNN（LeNet）时整理的概念"
date: 2026-07-20 00:00:00 +0900
last_modified_at: 2026-07-20 00:00:00 +0900
tags: [深度学习, numpy, cnn]
math: true
toc: true
lang: zh
translation_key: cnn-from-scratch
permalink: /zh/cnn-from-scratch/
---

这是一篇在仅使用 NumPy（不借助任何深度学习框架，手写前向/反向传播）实现用于 MNIST 手写数字分类的 CNN（LeNet 结构）过程中，整理的概念笔记。

原论文：[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - II. Convolutional Neural Networks For Isolated Character Recognition

## 结构
<img src="/assets/images/cnn-lenet-architecture.png" width="90%" class="align-center">

## 前向传播 (Forward Propagation)

### 卷积

<img src="/assets/images/cnn-convolution-layer.png" width="50%" class="align-center">

将一个较小的滤波器（卷积核）在输入上按 `stride` 逐步滑动，对重叠区域（patch）与卷积核逐元素相乘后求和（内积）的运算。一个滤波器在整幅图像上滑动扫描，从而找出出现特定模式（边缘、纹理等）的位置，其结果就构成了特征图（feature map）。

如果像全连接层（FC layer）那样，为每一对输入-输出各自分配独立的权重，参数数量会急剧膨胀，同时位置信息也会被忽略。卷积通过**在所有位置重复使用同一个滤波器（权重共享）**来减少参数数量，并通过只关注局部区域的**局部连接（local connectivity）**来保留图像的局部模式（空间结构）。

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} + 2 \cdot pad - kH}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} + 2 \cdot pad - kW}{stride} \right\rfloor + 1
\end{aligned}
$$

### 最大池化

<img src="/assets/images/cnn-pooling.png" width="50%" class="align-center">

将特征图划分为若干固定区间（窗口），在每个区间内只提取最大值，从而缩小特征图尺寸的运算。通过池化，信息被压缩后再传递给下一层。

池化之所以必要，主要有两个原因。
- **减少计算量/参数量**：缩小特征图尺寸，从而降低后续各层的计算量和内存占用。
- **对微小位置变化的鲁棒性（平移不变性，translation invariance）**：即使窗口内最大值所在的位置发生一两格的偏移，池化结果（最大值本身）通常也不会有太大变化，因此即便输入发生轻微平移或形变，模型也能给出相似的响应。

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} - pool}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} - pool}{stride} \right\rfloor + 1
\end{aligned}
$$

## 反向传播 (Backpropagation)

### 卷积的梯度
- `db`：将 dout 的总和传播到 db
- `dW`：遍历 patch * dout，累加传播到 dW
- `dX`：将 W[f] * dout 以 `+=` 的方式累加到对应的重叠 patch 位置上（由于同一输入区域被多个输出位置的计算重复使用，因此需要将梯度相加）

### 最大池化的梯度
在前向过程中，每个池化窗口内只有取得最大值的输入才对输出产生了影响，因此在反向过程中，只将梯度传递给前向过程中被选为最大值的那个位置，其余位置传递 0。

---

## im2col / col2im

### im2col 为何更快

基于 for 循环的实现会针对 batch(N) × 滤波器数量(C_out) × 输出位置(H_out × W_out) 的组合，在 Python 层级运行 for 循环，每次迭代都要将一个小 patch 与一个滤波器相乘后求和。此时，比起实际的乘加运算量，**Python 解释器的迭代开销**（类型检查、对象创建、函数调用等）所占的比重要大得多。

im2col 的做法是消除这种循环，转而
1. 预先把所有重叠的 patch 全部取出，展开为二维矩阵 `col`（patch 数量 × patch 大小）；
2. 将滤波器也展开为二维矩阵 `W_col`（patch 大小 × 滤波器数量）；
3. 通过**一次矩阵乘法** `col @ W_col`，同时计算出所有 patch-滤波器组合的内积。

矩阵乘法在 NumPy 内部会调用 BLAS（OpenBLAS、MKL 等）中高度优化的 C/Fortran 例程，这些例程利用了
- **SIMD 向量指令**：一次同时对多个元素进行乘加运算
- **对缓存友好的分块（blocking/tiling）**：将内存访问降到最低
- **多线程**：将运算分散到多个核心上

也就是说，即便计算量相同，Python 的 for 循环由于解释器开销，耗时会远高于纯计算时间，而矩阵乘法则几乎不受此开销影响，能够接近硬件本身的性能上限。

**权衡（trade-off）**：由于 patch 之间重叠的区域会被原样复制并存储（`col` 的大小约为原始输入的 `kH*kW` 倍），卷积核越大、stride 越小（重叠越多），内存占用就会显著增加。也就是说，im2col 是一种**用更多内存换取运算向量化**的方式。

## 速度比较

### for 循环 vs im2col

记录通过 `compare_conv_speed` 实测得到的数值。一般趋势如下：
- 批大小 N、通道数、输出尺寸（H_out × W_out）越大，for 循环的迭代次数就越多，与 im2col 之间的差距也就越大（通常相差数十到数百倍）。
- 反之，在输入非常小的情况下，构建 `col` 本身（reshape/transpose）所占的相对比重会变大，因此加速比可能会比较小。

### NumPy vs PyTorch (CPU)

虽然 NumPy 通过 im2col + 矩阵乘法也用到了 BLAS，但 PyTorch（`nn.Conv2d`）即便在 CPU 上通常也更快。
- PyTorch 使用专为深度学习优化的库（如 oneDNN 等），会根据具体情况自动选择更高效的方式，而不局限于 im2col+GEMM，例如直接卷积（direct convolution）或**Winograd 算法**（例如在 3x3 卷积核下本身就减少乘法次数）等。
- 不仅前向传播，反向传播（autograd）也是用 C++ 实现的，因此不像我们的实现那样存在在 Python 中传递 cache 的开销。
- 在计算图层面还应用了多线程并行化、内存复用、算子融合（kernel fusion）等优化手段。

也就是说，如果说 NumPy 的 im2col 只是"消除了 for 循环"，那么 PyTorch 则是在此基础上进一步加上了"针对卷积的专用算法选择 + 底层优化"。

### CPU vs GPU

GPU 在卷积运算上更具优势的根本原因在于硬件结构的差异。
- **CPU**：少量（通常几十个以内）性能强大的核心。针对复杂的分支预测和较大的缓存层级进行了优化，擅长处理顺序执行或分支较多的任务。
- **GPU**：拥有数千个结构简单的核心（SIMT：Single Instruction, Multiple Threads），专为将同一运算同时施加于大量数据而设计。

卷积是一种**易并行（embarrassingly parallel）**运算，输出中的每个 patch 都可以相互独立地计算，这与 GPU 的并行结构正好契合。此外：
- GPU 拥有相较 CPU 高得多的内存带宽（如 HBM 等），能够更快地读写大量数据。
- cuDNN 之类的库针对 GPU 实现了 Winograd、基于 FFT 的卷积等能够本身减少乘法次数的算法。
- 现代 GPU 具备诸如 **Tensor Core** 之类的专用硬件单元，用于加速矩阵乘法（尤其是 FP16/TF32 等低精度运算），使得 GEMM 运算本身速度大幅提升。

因此，在大规模卷积运算中，GPU 相较 CPU 常常能快数十到数百倍，但具体的倍数会因模型/批大小和硬件而有很大差异，需要通过实测来确认。
