---
layout: post
title: "Concepts I Learned Implementing a CNN (LeNet) From Scratch With NumPy"
date: 2026-07-20 00:00:00 +0900
last_modified_at: 2026-07-20 00:00:00 +0900
tags: [deep learning, numpy, cnn]
math: true
toc: true
lang: en
translation_key: cnn-from-scratch
permalink: /en/cnn-from-scratch/
---

Notes on concepts I put together while implementing a CNN (LeNet architecture) for MNIST handwritten digit classification using only NumPy — writing forward/backward propagation by hand, without a deep learning framework.

Original paper: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - II. Convolutional Neural Networks For Isolated Character Recognition

## Architecture
<img src="/assets/images/cnn-lenet-architecture.png" width="90%" class="align-center">

## Forward Propagation

### Convolution

<img src="/assets/images/cnn-convolution-layer.png" width="50%" class="align-center">

An operation that slides a small filter (kernel) over the input by `stride` at a time, multiplies the overlapping region (patch) with the kernel element-wise, and sums (dot product) the result. As a single filter sweeps across the whole image, it locates positions where a particular pattern (edge, texture, etc.) appears, and the result becomes a feature map.

If we connected every input-output pair with its own weight, as in an FC layer, the number of parameters would explode and positional information would be discarded. Convolution reduces the number of parameters by **reusing the same filter across every position (weight sharing)**, and preserves the local patterns (spatial structure) of the image through **local connectivity**, which only looks at a local region.

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} + 2 \cdot pad - kH}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} + 2 \cdot pad - kW}{stride} \right\rfloor + 1
\end{aligned}
$$

### Max Pooling

<img src="/assets/images/cnn-pooling.png" width="50%" class="align-center">

An operation that divides the feature map into fixed regions (windows) and extracts only the largest value from each region, reducing the size of the feature map. Through pooling, information is compressed before being passed to the next layer.

There are two main reasons pooling is needed.
- **Reducing computation/parameters**: shrinking the feature map size lowers the computation and memory usage of the layers that follow.
- **Robustness to small positional shifts (translation invariance)**: even if the position of the maximum value within a window shifts by a cell or two, the pooling result (the max value itself) tends not to change much, so the model responds similarly even when the input is slightly shifted or distorted.

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} - pool}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} - pool}{stride} \right\rfloor + 1
\end{aligned}
$$

## Backpropagation

### Gradient of Convolution
- `db`: propagate the sum of dout into db
- `dW`: iterate over patch * dout, accumulating into dW
- `dX`: accumulate W[f] * dout onto the overlapping patch positions with `+=` (since the same input region overlaps with the computation of multiple output positions, the gradients must be added together)

### Gradient of Max Pooling
In the forward pass, only the input holding the maximum value in each pooling window affected the output, so in the backward pass the gradient is passed only to the position that was selected as the maximum during the forward pass, while 0 is passed to all other positions.

---

## im2col / col2im

### Why im2col is fast

A for-loop based implementation runs a Python-level for loop over batch(N) × number of filters(C_out) × output positions(H_out × W_out), and on each iteration multiplies and sums one small patch with one filter. Here, **the interpreter overhead of Python's iteration** (type checking, object creation, function calls, etc.) dominates far more than the actual amount of multiply/add work.

Instead of this iteration, im2col
1. first extracts all the overlapping patches and unfolds them into a 2D matrix `col` (number of patches × patch size),
2. also unfolds the filter into a 2D matrix `W_col` (patch size × number of filters), then
3. computes the dot products of every patch-filter combination simultaneously with a **single matrix multiplication**, `col @ W_col`.

Under the hood, NumPy's matrix multiplication calls highly optimized C/Fortran routines from BLAS (OpenBLAS, MKL, etc.), and these routines take advantage of
- **SIMD vector instructions** to multiply/sum several elements at once
- **cache-friendly blocking/tiling** to minimize memory access
- **multithreading** to spread computation across multiple cores

In other words, for the same amount of computation, a Python for-loop takes far longer than the pure compute time due to interpreter overhead, while matrix multiplication gets close to the hardware's raw performance without that overhead.

**Trade-off**: because the overlapping regions of patches are copied and stored as-is (the size of `col` is roughly `kH*kW` times the original), memory usage increases significantly as the kernel gets larger and the stride gets smaller (i.e., as overlap increases). In short, im2col is a way of **trading more memory for vectorized computation**.

## Speed Comparison

### For-loop vs im2col

Recording values measured empirically with `compare_conv_speed`. General trends:
- The larger the batch size N, number of channels, and output size (H_out × W_out), the more iterations the for-loop needs, widening the gap with im2col (typically tens to hundreds of times).
- Conversely, for very small inputs, the relative cost of building `col` itself (reshape/transpose) becomes larger, so the speedup may turn out smaller.

### NumPy vs PyTorch (CPU)

Although NumPy also uses BLAS via im2col + matrix multiplication, PyTorch (`nn.Conv2d`) is usually still faster even on CPU.
- PyTorch uses libraries optimized specifically for deep learning (such as oneDNN), automatically choosing more efficient approaches depending on the situation — not just im2col+GEMM, but also direct convolution or the **Winograd algorithm** (e.g., reducing the actual number of multiplications for 3x3 kernels).
- Not only the forward pass but also the backward pass (autograd) is implemented in C++, so there is no overhead from passing caches back and forth in Python as in our implementation.
- At the computation-graph level, it applies multithreaded parallelization, memory reuse, kernel fusion, and more.

In short, if NumPy im2col is "removing the for-loop," PyTorch adds on top of that "choosing an algorithm specialized for convolution + low-level optimization."

### CPU vs GPU

The fundamental reason GPUs are advantageous for conv operations is the difference in hardware structure.
- **CPU**: a small number (usually a few dozen or fewer) of powerful cores. Optimized for complex branch prediction and large cache hierarchies, it excels at sequential or branch-heavy workloads.
- **GPU**: thousands of simple cores (SIMT: Single Instruction, Multiple Threads), specialized for applying the same operation to a large amount of data simultaneously.

Convolution is an **embarrassingly parallel** operation where each output patch can be computed independently of the others, which lines up exactly with the GPU's parallel structure. In addition,
- GPUs have far higher memory bandwidth than CPUs, through things like HBM, letting them read and write large volumes of data quickly.
- Libraries such as cuDNN implement algorithms like Winograd and FFT-based convolution, which reduce the actual number of multiplications, tailored specifically for the GPU.
- Modern GPUs have units like **Tensor Cores** that accelerate matrix multiplication (especially at lower precisions such as FP16/TF32) with dedicated hardware, making the GEMM operation itself much faster.

As a result, for large-scale conv operations, GPUs are commonly tens to hundreds of times faster than CPUs, but the exact factor varies greatly depending on model/batch size and hardware, so it needs to be measured empirically.
