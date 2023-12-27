---
layout: post
title:  "Make-A-Video"
categories: 논문리뷰
date: 2023-06-23 11:40:18 +0900
tags: Diffusion 생성모델 Meta Video
mathjax: True
author: Haribo
---
* content
{:toc}
**Full Citation**: "Singer, Uriel, et al. "Make-a-video: Text-to-video generation without text-video data." arXiv preprint arXiv:2209.14792 (2022)."\
**Link to Paper**: [https://arxiv.org/abs/2209.14792](https://arxiv.org/abs/2209.14792) \
**Conference Details**: ICLR 2023 \
**Project Page**: [Link](https://makeavideo.studio/)

---

>* 선행 Text-to-Video 연구들은 다수의 video-text pair 데이터셋이 필요했으나, 사전학습 된 Diffusion 모델의 능력을 활용해 video-text 없이 video 데이터만을 활용해 고퀄리티 text-to-video 생성모델 학습 방식을 선보임. 
>* 4D 입력인 video 처리를 위해 Spatial/Temporal Convolution + Attention 연산을 활용.



<div style="text-align: center;">   
  <figure>     
    <img src="https://makeavideo.studio/assets/overview.webp">     
  </figure> 
  <figcaption>A dog wearing a Superhero outfit with red cape flying through the sky</figcaption>   
</div>




# 1. Introduction

**Motivation**

* 인터넷상 수십억개의 image-text 쌍 데이터셋으로 Text-to-Image(T2I) 모델은 비약적으로 발전.
* 하지만 video-text 쌍 데이터셋은 많이 없어서 Text-to-Video(T2V) 모델은 발전이 더뎠음.
* *이미 잘 학습된 T2I 모델을 활용해 T2V 모델을 만들면 어떨까?* 이게 저자들의 생각.
  * T2V 모델을 처음부터 학습시키는 건 비효율적이다.
  * 잘 학습된 T2I 모델을 활용하면 text label이 없는 video만으로도 unsupervised 방식으로 finetuning이 가능하다.

**결론**

* 한 줄의 텍스트로 긴 비디오를 설명하기엔 부족하지만, 짧은 비디오는 한 줄의 텍스트로도 충분하다.
  * 관련 선행연구: image-based action recognition systems(Girish et al., 2020)
* 기존 T2I 모델은 공간적 정보만 다루므로, 네트워크를 확장해 시간 정보도 다루는 spatial-temporal 네트워크를 구성했다.
* 이를 통해 기존 T2I에서 새로운 T2V 네트워크로의 전환을 가능하게 했다.
* Visual quality와 부드러움을 개선하기 위해 super-resolution 모델과 frame interpolation 모델도 추가로 학습했다.

**Contribution**

* 기존의 T2I 모델에 시공간연산 확장을 통해 효율적인 T2V 모델, Make-A-Video, 모델 제안.
* text-image 를 활용했기 때문에 text-video 데이터셋이 필요없고, 대량의 unlabeled video 데이터셋으로 unsupervised training 가능.
* User로부터 주어진 text 입력에 대해 최초로 시공간적을 고려한 super-resolution 방법 제안.
* T2V 분야에서 quantitative, qualitative 평가에서 SOTA 달성.



































