---
layout: post
title: What are 3 important object detection models you should know?
date: 2025-01-31 17:25 +0800
last_modified_at: 2025-01-31 17:25 +0800
tags: [object detection]
toc:  true
---

Object detection is one of the most actively researched topic.This field is applied across various industries, from detecting tumors in the medical field to identifying specific vehicles in the defense sector. This post explains three important object detection models called RCNN, YOLO, DETR.

---

## RCNN

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

**RCNN** (Region-based Convolutional Neural Networks) is a two-stage object detection model that first generates possible object regions in an image using **selective search**, and then classifies each region using a **CNN** and an **SVM**. It improves accuracy by refining the object’s position with **bounding box regression**. However, since it involves two stages, the model is slow and not suitable for real-time applications. Models like Fast RCNN and Faster RCNN have improved on these issues.

![Figure 1](/assets/images/object_detection_01.png)

---

## YOLO

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

**YOLO** is the first one-stage object detection model to predict bounding boxes and class probabilities simultaneously. The image is divided into **a grid of cells**. Each grid cell predicts bounding boxes, object classes, and confidence scores for objects within the cell. This model strikes a balance between accuracy and efficiency, making it widely used for real-time detection. YOLO is currently being developed by Ultralytics.

![Figure 2](/assets/images/object_detection_02.png)

---

## DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

Transformers were initially developed for natural language processing, but their power sparked active research into applying them to computer vision models. **DETR** is a result of this. The image is passed through a **CNN** to obtain feature maps. A transformer network uses **self-attention** to focus on different parts of the image. It predicts the locations and classes of objects through the transformer output. It is particularly effective at detecting small objects and handling complex situations.

![Figure 3](/assets/images/object_detection_03.png)

---

## Comparison

|Feature|RCNN|YOLO|DETR|
|-----|---|---|---|
|Model Type|Region Proposal + CNN|End-to-End Grid-based|End-to-End Transformer-based|
|Speed|Slow|Fast|Slow|
|Accuracy|High|Medium-High|High|
|Object Detection|Region proposals|CNN processing|Single pass prediction|Transformer-based attention|
|Advantages|High accuracy, detailed detection|Real-time speed, simple structure|Good for small objects, handles complex relationships well|
|Disadvantages|Slow speed, high computation|Lower accuracy in some cases, struggles with small objects|Slow training & inference, needs lots of data|