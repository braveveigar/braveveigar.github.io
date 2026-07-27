---
layout: post
title: "知っておくべき3つの重要な物体検出モデル"
date: 2025-01-31 17:25 +0800
last_modified_at: 2025-01-31 17:25 +0800
tags: [物体検出]
toc: true
lang: ja
translation_key: object-detection
permalink: /ja/object-detection/
---

物体検出(Object Detection)は、現在最も活発に研究が進められている分野の一つです。医療分野における腫瘍の検出から、国防分野における特定車両の識別まで、さまざまな産業で応用されています。本記事では、RCNN、YOLO、DETRという3つの重要な物体検出モデルについて解説します。

---

## RCNN

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

**RCNN**(Region-based Convolutional Neural Networks)は、まず**selective search**を用いて画像内に物体が存在する可能性のある領域を生成し、その後各領域を**CNN**と**SVM**で分類する2段階(two-stage)の物体検出モデルです。**bounding box regression**によって物体の位置を補正することで精度を高めています。ただし、2段階の処理を要するため速度が遅く、リアルタイム用途には向いていません。この問題はFast RCNNやFaster RCNNといった後続モデルで改善されています。

![Figure 1](/assets/images/object_detection_01.png)

---

## YOLO

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

**YOLO**は、バウンディングボックスとクラス確率を同時に予測する最初の1段階(one-stage)物体検出モデルです。画像は**グリッド(grid)状のセル**に分割され、各グリッドセルはそのセル内にある物体のバウンディングボックス、クラス、信頼度スコア(confidence score)を予測します。精度と効率のバランスに優れているため、リアルタイム検出に広く利用されています。YOLOは現在Ultralyticsによって開発が続けられています。

![Figure 2](/assets/images/object_detection_02.png)

---

## DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

Transformerはもともと自然言語処理のために開発されましたが、その高い性能からコンピュータビジョン分野への応用研究が活発に行われるようになりました。**DETR**はその成果の一つです。画像はまず**CNN**を通して特徴マップ(feature map)を得ます。次にTransformerネットワークが**self-attention**を用いて画像のさまざまな部分に注目し、その出力を通じて物体の位置とクラスを予測します。特に小さな物体の検出や複雑な状況の処理に優れています。

![Figure 3](/assets/images/object_detection_03.png)

---

## Comparison

|項目|RCNN|YOLO|DETR|
|-----|---|---|---|
|モデルタイプ|Region Proposal + CNN|End-to-End グリッドベース|End-to-End Transformerベース|
|速度|遅い|速い|遅い|
|精度|高い|中〜高|高い|
|長所|高精度で詳細な検出が可能|リアルタイムの速度、シンプルな構造|小さな物体の検出に強く、複雑な関係もうまく処理できる|
|短所|速度が遅く、計算コストが高い|場合によっては精度が低下し、小さな物体の検出が苦手|学習・推論に時間がかかり、大量のデータが必要|
