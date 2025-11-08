---
layout: post
title:  "Accelerater 풀 스파링 (2편, 분산학습)"
categories: 실습
date: 2025-11-08 11:40:18 +0900
tags: AI Optimizer 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

분산학습이 도대체 뭔지 아직도 사실 잘 모른다.  GPU 여러개에 모델 복사해 놓고 backprop 할 때 gradiend만 취합해서 보내는건지, GPU 여러개를 하나처럼 쓰는건지, 어떻게 쓰는건지 등등.  
이참에 확실히 정리할 예정.

# 분산 학습

| 구분  | 개념 |                       
|-------|----|
|Single GPU | 한 GPU로 학습  |
| Data Parallel (데이터 병렬) | 모델 복사 N개 → 각 GPU가 다른 미니배치 학습 | 
| Model Parallel (모델 병렬)| 모델 자체를 여러 GPU에 나눠서 저장 |
|Pipeline Parallel| 모델을 여러 파트로 나누고 파이프라인처럼 순차 실행 |
|Tensor Parallel| 한 레이어 내부 연산을 여러 GPU가 나눠 처리 |
| Multi-Node (분산 학습)| GPU 여러 개가 여러 서버(노드) 에 흩어져 있음 |
|Hybrid Parallel| 위 병렬 방식을 혼합  |


