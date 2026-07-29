---
layout: post
title: CCTV-로봇 지도 좌표 변환(Homography)과 Re-ID 다중 카메라 테스트
date: 2026-07-29 00:00:00 +0900
last_modified_at: 2026-07-29 00:00:00 +0900
tags: [homography, re-id, deepstream, slam]
math: true
toc: true
lang: ko
translation_key: cctv-robot-map-reid
permalink: /cctv-robot-map-reid/
---

AI CCTV가 포착한 위치로 로봇을 정확히 출동시킬 수 있을까? SLAM 지도와 CCTV 화면, 서로 다른 두 세계를 잇는 다리를 Homography로 놓아봤다.

## 배경

로봇은 이미 SLAM으로 만든 지도 위에서 자기 위치를 정확히 파악하고 있었다. 반면 CCTV는 그저 자신이 보고 있는 화면 안에서 "여기에 뭔가 있다"는 바운딩 박스 하나만 던져줄 뿐이었다. 카메라의 픽셀 좌표와 로봇이 쓰는 지도 좌표는 애초에 서로 다른 세계였다. 이 둘을 이어 줄 다리가 필요했고, 그 답은 Homography였다.

## Homography로 CCTV-로봇 지도 좌표 맞추기

Homography는 한 평면을 다른 평면으로 대응시키는 $3\times3$ 변환 행렬이다. CCTV 화면(카메라 이미지 평면)의 한 점 $(x, y)$를 로봇 지도(월드 평면)의 좌표로 옮기고 싶을 때 다음과 같이 쓸 수 있다.

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

여기서 나온 동차좌표 $(x', y', w')$를 정규화하면 실제 지도 좌표를 얻는다.

$$
(x_{map},\ y_{map}) = \left(\frac{x'}{w'},\ \frac{y'}{w'}\right)
$$

$H$는 CCTV 화면과 로봇 지도 양쪽에서 서로 대응하는 점을 4쌍 이상 짝지어주면 계산할 수 있다. 예를 들어 CCTV 화면 속 바닥 위 특징점 4곳과, 로봇 지도 상에서 같은 위치에 해당하는 좌표 4곳을 짝지어 추정하는 식이다. 이렇게 구한 $H$가 있으면, 이후로는 CCTV 화면에 사건이 잡힐 때마다 그 픽셀 좌표를 바로 로봇 지도 좌표로 변환할 수 있다.

<img src="/assets/images/homography_test.png" width="50%" class="align-center">

## 여러 대의 CCTV 연동과 Re-ID 테스트

여러 대의 CCTV를 연동한 김에, 위치 정보뿐 아니라 특정 인물을 계속 추적하거나 특정 물건을 빠르게 찾아내는 것도 가능한지 궁금해져서 Re-ID(Re-Identification)까지 함께 테스트해봤다.

### 사용한 모델

- DeepStream 7.1
- FGIE(1차 검출): YOLO26n
- SGIE(Re-ID 임베딩 추출): MSMT17로 학습한 CLIP-ReID+SIE+OLP — [CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels](https://arxiv.org/abs/2211.13977)

### Re-ID 아이디어

Re-ID 모델은 바운딩 박스 안에 잡힌 객체의 외형 특징을 학습해서, 그 특징들을 하나의 임베딩 벡터로 압축해 출력한다. 서버 쪽에서는 각 카메라에서 들어오는 임베딩들을 모아두고, 새로 들어온 임베딩과 기존에 추적 중이던 임베딩들 사이의 코사인 유사도를 계산한다. 유사도가 특정 임계값(예: 0.6) 이상이면 동일 객체로 판단하고 같은 global ID를 부여하는 방식이다.

<img src="/assets/images/reid.png" width="50%" class="align-center">

## 테스트 영상

<video controls width="100%">
  <source src="/assets/videos/homography-reid-multicam-test.mp4" type="video/mp4">
</video>

화면 왼쪽은 CCTV 한 대가 보고 있는 장면이고, 오른쪽은 SLAM으로 만든 사무실 지도다. 두 대의 카메라에서 검출한 사람의 bounding box 좌표를 MQTT로 받아 Homography로 변환한 뒤, 지도 위에 점으로 찍어 표시했다. 동일한 global ID로 판단되면 같은 색 점으로 표시된다.

## 관찰: 뒤돌자마자 바뀐 Global ID

영상에서 첫 global ID(`10681249`)가 두 번째 CCTV로도 그대로 이어지는 것처럼 보였지만, 인물이 몸을 돌리는 순간 곧바로 새로운 global ID로 바뀌는 것을 확인했다. 앞모습과 뒷모습, 조명 차이만으로도 임베딩 값이 크게 달라진다는 뜻이다. 벤치마크 데이터셋(MSMT17)에 최적화된 모델이 실제 현장 조건에서는 필요한 만큼 강건한 임베딩을 내놓지 못하는 걸 보여주는 사례로 보인다.

## 테스트 결과

**Re-ID**: 하체가 가려지거나 뒤를 도는 등 외형이 조금만 달라져도 ID가 자주 변경됐다. 통제된 데이터셋과 달리 **실제 CCTV 환경에서는 가림, 각도, 해상도 등의 영향이 커**서, 단독으로 상용 적용하기엔 아직 불안정해 보인다.

**좌표 변환**: 영상과 지도에 기준점을 정확히 지정하면 위치가 전반적으로 잘 표시됐다. 다만 **평지이고 사람의 발 위치가 보인다**는 전제가 있으며, 광각·원거리·가림 상황에서는 오차가 커질 수 있다.

좌표 변환은 사람 위치 표시 외에도 CCTV 감시 영역, 중복 영역, 사각지대 시각화에 활용할 수 있을 것으로 보인다.
