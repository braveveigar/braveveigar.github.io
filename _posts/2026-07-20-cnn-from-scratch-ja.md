---
layout: post
title: "NumPyでCNN（LeNet）をゼロから実装しながら整理した概念"
date: 2026-07-20 00:00:00 +0900
last_modified_at: 2026-07-20 00:00:00 +0900
tags: [深層学習, numpy, cnn]
math: true
toc: true
lang: ja
translation_key: cnn-from-scratch
permalink: /ja/cnn-from-scratch/
---

MNISTの手書き数字分類のためのCNN（LeNet構造）を、NumPyのみで（ディープラーニングフレームワークを使わずforward/backward propagationを自前で実装して）作りながら整理した概念ノートです。

原論文: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - II. Convolutional Neural Networks For Isolated Character Recognition

## 構造
<img src="/assets/images/cnn-lenet-architecture.png" width="90%" class="align-center">

## 順伝播 (Forward Propagation)

### 畳み込み

<img src="/assets/images/cnn-convolution-layer.png" width="50%" class="align-center">

小さなフィルタ（カーネル）を入力上で `stride` の分だけずらしながら移動させ、重なった領域（patch）とカーネルを要素ごとに掛け合わせてから合算（内積）する演算。1つのフィルタが画像全体を走査することで、特定のパターン（エッジ、テクスチャなど）が存在する位置を見つけ出し、その結果がfeature mapとなる。

FC layerのようにすべての入力-出力のペアをそれぞれ別の重みで結合すると、パラメータ数が爆発的に増え、位置情報も無視されてしまう。畳み込みは**同一のフィルタを全位置で再利用する（重み共有）**ことでパラメータ数を減らし、局所的な領域のみを見る**local connectivity**によって画像の局所的なパターン（空間構造）を保持する。

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} + 2 \cdot pad - kH}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} + 2 \cdot pad - kW}{stride} \right\rfloor + 1
\end{aligned}
$$

### マックスプーリング

<img src="/assets/images/cnn-pooling.png" width="50%" class="align-center">

feature mapを一定の区間（window）に分割し、各区間の中で最大の値のみを取り出すことでfeature mapのサイズを縮小する演算。プーリングによって情報が圧縮された上で次のlayerへと伝えられる。

プーリングが必要な理由は大きく2つある。
- **計算量・パラメータ数の削減**: feature mapのサイズを縮小することで、後続のlayerの計算量とメモリ使用量を下げる。
- **微小な位置変化に対する頑健性（平行移動不変性、translation invariance）**: 区間内での最大値の位置が1〜2マス移動しても、プーリングの結果（最大値そのもの）はあまり変わらないため、入力がわずかに移動・歪んでもモデルは同じように反応できる。

$$
\begin{aligned}
H_{out} &= \left\lfloor \frac{H_{in} - pool}{stride} \right\rfloor + 1 \\
W_{out} &= \left\lfloor \frac{W_{in} - pool}{stride} \right\rfloor + 1
\end{aligned}
$$

## 逆伝播 (Backpropagation)

### 畳み込みの勾配
- `db`: doutの合計をdbに伝播する
- `dW`: patch * doutを巡回しながらdWに累積して伝播する
- `dX`: W[f] * doutを、重なり合うpatchの位置に `+=` で累積する（同じ入力領域が複数の出力位置の計算に重なって使われているため、勾配を足し合わせる必要がある）

### マックスプーリングの勾配
順伝播の過程では、各プーリングwindow内で最大値を持つ入力のみが出力に影響を与えたため、逆伝播の過程では順伝播で最大値として選ばれた位置にのみ勾配を伝え、それ以外の位置には0を伝える。

---

## im2col / col2im

### im2colが速い理由

for文ベースの実装は、batch(N) × フィルタ数(C_out) × 出力位置(H_out × W_out) の分だけPythonレベルのforループを回し、繰り返しのたびに小さなpatch1つとフィルタ1つを掛け合わせて合算する。このとき、実際の乗算・加算量よりも**Pythonインタプリタの反復オーバーヘッド**（型チェック、オブジェクト生成、関数呼び出しなど）の方がはるかに大きく効いてくる。

im2colはこの繰り返しをなくす代わりに、
1. 重なり合うpatchをすべて先に取り出して2次元行列 `col`（patch数 × patchサイズ）に展開し、
2. フィルタも2次元行列 `W_col`（patchサイズ × フィルタ数）に展開した上で、
3. `col @ W_col` の**行列積1回**で、すべてのpatch-フィルタの組み合わせの内積を同時に計算する。

行列積はNumPy内部でBLAS（OpenBLAS、MKLなど）の高度に最適化されたC/Fortranルーチンを呼び出しており、このルーチンは
- **SIMDベクトル命令**により複数の要素を一度に乗算・合算し
- **キャッシュに親和的なblocking/tiling**によりメモリアクセスを最小化し
- **マルチスレッド**により演算を複数のコアに分散する

といった手法を活用している。つまり同じ演算量であっても、Pythonのfor文はインタプリタのオーバーヘッドのために純粋な計算時間よりはるかに長くかかるのに対し、行列積はそのオーバーヘッドなしにハードウェア本来の性能に近づくことができる。

**トレードオフ**: patch同士が重なる領域をそのままコピーして保存するため（`col`のサイズは元のデータに対しておおよそ `kH*kW` 倍）、カーネルが大きいほど・strideが小さいほど（重なりが多いほど）メモリ使用量が大きく増加する。つまりim2colは**メモリをより多く使うことで演算をベクトル化する**方式である。

## 速度比較

### for文 vs im2col

`compare_conv_speed` で実測した値を記録する。一般的な傾向としては次の通り。
- バッチサイズN、チャンネル数、出力サイズ（H_out × W_out）が大きいほど、for文の反復回数が増え、im2colとの差が大きくなる（多くの場合数十〜数百倍の差になる）。
- 逆に非常に小さい入力では、`col` を作るreshape/transpose自体の相対的な比重が大きくなり、speedupが小さくなることがある。

### NumPy vs PyTorch（CPU）

im2col + 行列積によりNumPyもBLASを使っているが、PyTorch（`nn.Conv2d`）はCPU上でも通常はより速い。
- PyTorchはディープラーニング専用に最適化されたライブラリ（oneDNNなど）を使用しており、状況に応じてim2col+GEMMではなくdirect convolutionや**Winogradアルゴリズム**（例: 3x3カーネルにおいて乗算回数自体を減らす）など、より効率的な方式を自動的に選択する。
- 順伝播だけでなく逆伝播（autograd）までC++で実装されているため、我々の実装のようにPython側でcacheをやり取りするオーバーヘッドがない。
- 計算グラフのレベルでマルチスレッド並列化、メモリの再利用、カーネルフュージョン（融合）などを適用している。

つまり、NumPyのim2colが「for文をなくしたもの」だとすれば、PyTorchは「畳み込みに特化したアルゴリズム選択+低レベル最適化」までを加えたものと言える。

### CPU vs GPU

GPUがconv演算に有利な根本的な理由は、ハードウェア構造の違いにある。
- **CPU**: 少数（数十個以下）の強力なコア。複雑な分岐予測や大きなキャッシュ階層に最適化されており、逐次的、あるいは分岐の多い処理に強い。
- **GPU**: 数千個の単純なコア（SIMT: Single Instruction, Multiple Threads）を持ち、同一の演算を大量のデータに同時に適用することに特化している。

畳み込みは、出力の各patchが互いに独立して計算できる**embarrassingly parallel**な演算であり、GPUの並列構造とまさに合致する。さらに、
- GPUはHBMなど、CPUに比べてはるかに高いメモリ帯域幅を持ち、大量のデータを高速に読み書きできる。
- cuDNNのようなライブラリは、Winograd、FFTベースの畳み込みなど、乗算回数自体を減らすアルゴリズムまでGPU向けに実装している。
- 最新のGPUは**Tensor Core**のような、行列積（特にFP16/TF32など低精度）を専用ハードウェアで高速化するユニットを備えており、GEMM演算自体がはるかに高速になる。

結果として、大規模なconv演算においてGPUはCPUに比べて数十〜数百倍速いケースが多いが、正確な倍率はモデル・バッチサイズやハードウェアによって大きく変わるため、実測が必要である。
