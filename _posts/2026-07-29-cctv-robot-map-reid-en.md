---
layout: post
title: "CCTV-to-Robot Map Coordinate Transform (Homography) and a Multi-Camera Re-ID Test"
date: 2026-07-29 00:00:00 +0900
last_modified_at: 2026-07-29 00:00:00 +0900
tags: [homography, re-id, deepstream, slam]
math: true
toc: true
lang: en
translation_key: cctv-robot-map-reid
permalink: /en/cctv-robot-map-reid/
---

Can a robot be dispatched precisely to wherever an AI CCTV camera spots something? I ended up bridging two very different worlds — a SLAM map and a CCTV feed — with Homography.

## Background

The robot already knew exactly where it was, thanks to a map built with SLAM. The CCTV, on the other hand, could only hand over a bounding box saying "something's here" within its own view. The camera's pixel coordinates and the robot's map coordinates were simply two different worlds from the start. Bridging them was the whole problem, and the answer was Homography.

## Matching CCTV and Robot Map Coordinates With Homography

Homography is a $3\times3$ transformation matrix that maps one plane onto another. To move a point $(x, y)$ on the CCTV screen (the camera image plane) to a coordinate on the robot's map (the world plane), it can be written as:

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

Normalizing the resulting homogeneous coordinates $(x', y', w')$ gives the actual map coordinate.

$$
(x_{map},\ y_{map}) = \left(\frac{x'}{w'},\ \frac{y'}{w'}\right)
$$

$H$ can be solved for once you have at least 4 pairs of corresponding points between the CCTV screen and the robot map — for example, pairing 4 feature points on the floor in the CCTV view with their matching coordinates on the robot map. Once $H$ is known, any pixel coordinate where an event is captured on the CCTV screen can be converted directly into a robot map coordinate.

<img src="/assets/images/homography_test.png" width="50%" class="align-center">

## Linking Multiple CCTVs and Testing Re-ID

Since I already had multiple CCTVs linked together, I got curious whether I could go beyond just location — keep tracking a specific person, or quickly locate a specific object — so I tested Re-ID (Re-Identification) alongside it.

### Models Used

- DeepStream 7.1
- FGIE (primary detector): YOLO26n
- SGIE (Re-ID embedding extractor): CLIP-ReID+SIE+OLP trained on MSMT17 — [CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels](https://arxiv.org/abs/2211.13977)

### The Idea Behind Re-ID

A Re-ID model learns the appearance features of whatever's inside a bounding box, and outputs those features compressed into a single embedding vector. On the server side, incoming embeddings from each camera are collected, and the cosine similarity between a new embedding and the embeddings already being tracked is computed. If the similarity is above a certain threshold (e.g. 0.6), it's judged to be the same object and assigned the same global ID.

<img src="/assets/images/reid.png" width="50%" class="align-center">

## Test Video

<video controls width="100%">
  <source src="/assets/videos/homography-reid-multicam-test.mp4" type="video/mp4">
</video>

The left side of the frame is the live view from one CCTV camera, and the right side is the office map built with SLAM. Bounding-box coordinates for people detected by both cameras are received over MQTT, transformed with Homography, and plotted as dots on the map. Dots sharing the same color mean they were judged to be the same global ID.

## Observation: The Global ID That Changed the Moment He Turned Around

In the video, the first global ID (`10681249`) seemed to carry over to the second CCTV, but the moment the person turned around, it immediately switched to a new global ID. In other words, facing forward vs. backward, or even just a change in lighting, was enough to shift the embedding significantly. It looks like a case showing that a model optimized for a benchmark dataset (MSMT17) doesn't yet produce embeddings robust enough for real-world conditions.

## Test Results

**Re-ID**: Even a small change in appearance — legs getting occluded, turning around — caused the ID to switch frequently. Unlike a controlled dataset, occlusion, angle, and resolution have a much bigger effect in a real CCTV environment, so it still looks too unstable to deploy on its own in production.

**Coordinate transform**: Once reference points are set accurately between the footage and the map, positions were displayed well overall. That said, it assumes flat ground and a visible foot position, and error can grow under wide-angle, long-distance, or occluded conditions.

Beyond just showing a person's location, coordinate transform also looks useful for visualizing CCTV coverage areas, overlapping coverage, and blind spots.
