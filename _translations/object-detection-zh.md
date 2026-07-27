---
layout: post
title: "你应该了解的3个重要目标检测模型"
date: 2025-01-31 17:25 +0800
last_modified_at: 2025-01-31 17:25 +0800
tags: [目标检测]
toc: true
lang: zh
translation_key: object-detection
permalink: /zh/object-detection/
---

目标检测(Object Detection)是当前研究最活跃的领域之一。从医疗领域的肿瘤检测到国防领域的特定车辆识别,该技术已被广泛应用于各行各业。本文将介绍三种重要的目标检测模型:RCNN、YOLO 和 DETR。

---

## RCNN

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

**RCNN**(Region-based Convolutional Neural Networks,基于区域的卷积神经网络)是一种两阶段(two-stage)目标检测模型。它首先使用**selective search**在图像中生成可能包含目标的候选区域,然后通过**CNN**和**SVM**对每个区域进行分类,并通过**边界框回归(bounding box regression)**修正目标的位置,从而提高准确率。但由于需要经过两个阶段,该模型速度较慢,不适合实时应用。Fast RCNN 和 Faster RCNN 等后续模型对这些问题进行了改进。

![Figure 1](/assets/images/object_detection_01.png)

---

## YOLO

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

**YOLO** 是第一个同时预测边界框和类别概率的单阶段(one-stage)目标检测模型。图像被划分为**网格(grid)状的单元格**,每个网格单元都会预测该单元内目标的边界框、类别以及置信度分数(confidence score)。该模型在准确率和效率之间取得了良好的平衡,因此被广泛用于实时检测场景。YOLO 目前由 Ultralytics 持续开发维护。

![Figure 2](/assets/images/object_detection_02.png)

---

## DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

Transformer 最初是为自然语言处理而设计的,但其强大的性能引发了将其应用于计算机视觉模型的研究热潮,**DETR** 正是这一研究方向的成果之一。图像首先经过**CNN**得到特征图(feature map),随后 Transformer 网络利用**self-attention**机制关注图像的不同区域,并通过 Transformer 的输出预测目标的位置和类别。该模型在检测小目标和处理复杂场景方面表现尤为出色。

![Figure 3](/assets/images/object_detection_03.png)

---

## Comparison

|特性|RCNN|YOLO|DETR|
|-----|---|---|---|
|模型类型|区域候选 + CNN|端到端网格式|端到端 Transformer 式|
|速度|慢|快|慢|
|准确率|高|中高|高|
|优点|准确率高,检测精细|速度快(实时),结构简单|擅长检测小目标,能很好地处理复杂关系|
|缺点|速度慢,计算量大|某些情况下准确率较低,难以检测小目标|训练和推理速度慢,需要大量数据|
