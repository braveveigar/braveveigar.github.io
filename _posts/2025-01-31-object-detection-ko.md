---
layout: post
title: "꼭 알아야 할 객체 탐지 모델 3가지"
date: 2025-01-31 17:25 +0800
last_modified_at: 2025-01-31 17:25 +0800
tags: [객체 탐지]
toc: true
lang: ko
translation_key: object-detection
permalink: /object-detection/
---

객체 탐지(Object Detection)는 현재 가장 활발히 연구되고 있는 분야 중 하나입니다. 의료 분야에서 종양을 탐지하는 것부터 국방 분야에서 특정 차량을 식별하는 것까지, 다양한 산업 전반에 걸쳐 활용되고 있습니다. 이 글에서는 RCNN, YOLO, DETR이라는 세 가지 중요한 객체 탐지 모델을 소개합니다.

---

## RCNN

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

**RCNN**(Region-based Convolutional Neural Networks)은 **selective search**를 이용해 이미지 내에서 객체가 존재할 가능성이 있는 영역을 먼저 생성한 뒤, 각 영역을 **CNN**과 **SVM**으로 분류하는 2단계(two-stage) 객체 탐지 모델입니다. **bounding box regression**을 통해 객체의 위치를 보정함으로써 정확도를 높입니다. 다만 두 단계를 거쳐야 하기 때문에 속도가 느려 실시간 애플리케이션에는 적합하지 않습니다. 이러한 문제는 Fast RCNN, Faster RCNN과 같은 후속 모델에서 개선되었습니다.

![Figure 1](/assets/images/object_detection_01.png)

---

## YOLO

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

**YOLO**는 바운딩 박스(bounding box)와 클래스 확률을 동시에 예측하는 최초의 1단계(one-stage) 객체 탐지 모델입니다. 이미지는 **격자(grid) 형태의 셀**로 분할되며, 각 그리드 셀은 해당 셀 내 객체에 대한 바운딩 박스, 클래스, 신뢰도 점수(confidence score)를 예측합니다. 정확도와 효율성 사이의 균형이 뛰어나 실시간 탐지에 널리 사용됩니다. YOLO는 현재 Ultralytics에서 개발을 이어가고 있습니다.

![Figure 2](/assets/images/object_detection_02.png)

---

## DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

트랜스포머(Transformer)는 원래 자연어 처리를 위해 개발되었지만, 그 강력한 성능 덕분에 컴퓨터 비전 모델에 적용하려는 연구가 활발히 진행되었습니다. **DETR**은 그 결과물 중 하나입니다. 이미지는 **CNN**을 거쳐 특징 맵(feature map)을 얻습니다. 이후 트랜스포머 네트워크가 **self-attention**을 이용해 이미지의 여러 부분에 주목합니다. 트랜스포머의 출력을 통해 객체의 위치와 클래스를 예측하며, 특히 작은 객체를 탐지하거나 복잡한 상황을 처리하는 데 뛰어난 성능을 보입니다.

![Figure 3](/assets/images/object_detection_03.png)

---

## Comparison

|항목|RCNN|YOLO|DETR|
|-----|---|---|---|
|모델 유형|Region Proposal + CNN|End-to-End Grid 기반|End-to-End Transformer 기반|
|속도|느림|빠름|느림|
|정확도|높음|중상|높음|
|장점|높은 정확도, 정밀한 탐지|실시간 속도, 단순한 구조|작은 객체 탐지에 강함, 복잡한 관계 처리에 우수|
|단점|느린 속도, 높은 연산량|경우에 따라 정확도가 낮고 작은 객체 탐지에 취약|느린 학습 및 추론 속도, 많은 데이터 필요|
