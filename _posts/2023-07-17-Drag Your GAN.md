---
layout: post
title:  "Drag Your GAN"
categories: 논문리뷰
date: 2023-07-17 11:40:18 +0900
tags: GAN 생성모델
mathjax: True
author: Haribo
---
* content
{:toc}
**Full Citation**: "Pan, Xingang, et al. "Drag your gan: Interactive point-based manipulation on the generative image manifold." ACM SIGGRAPH 2023 Conference Proceedings. 2023."\
**Link to Paper**: [https://arxiv.org/abs/2305.10973](https://arxiv.org/abs/2305.10973) \
**Conference Details**: ACM SIGGRAPH 2023

---

>* 새로운 GAN 모델을 만드는 것이 아닌 기존의 GAN (StyleGAN2)을 컨트롤 하는 연구.
>* Src, Tgt 두 종류의 포인터로 생성 된 이미지의 pose, shape, expression 등등을 변형.
>* GAN을 컨트롤 하는데에 있어 추가적인 인공지능 모델 학습이나 활용 필요없이 내부 featuremap domain에서 연산이 진행됨.
>* GAN의 잠재능력을 극한으로 활용하는 느낌.

<div style="text-align: center;">   
  <figure>     
    <img src="https://github.com/XingangPan/DragGAN/raw/main/DragGAN.gif">     
  </figure> 
</div>

**Web Demos**

[![Open in OpenXLab](https://camo.githubusercontent.com/a28f1d6dc75b31d084cb14e4c6f5bbd0f97bfbecd7bf260fac8dbd95df1c9430/68747470733a2f2f63646e2d7374617469632e6f70656e786c61622e6f72672e636e2f6170702d63656e7465722f6f70656e786c61625f6170702e737667)](https://openxlab.org.cn/apps/detail/XingangPan/DragGAN)

[![Huggingface](https://camo.githubusercontent.com/3cc55bf2ba0bc623f8f32e700b9834d27767be50da0a720d5a7f083768be244d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d4472616747414e2d6f72616e6765)](https://huggingface.co/spaces/radames/DragGan)



# 1. Introduction

생성 모델을 real-world applications에서 활용하기 위해선 이미지 생성을 유저가 원하는대로 컨트롤 할 수 있어야한다. 이를 위해 다양한 conditional image synthesis 방법론들이 등장했는데 대표적으로 semantic map을 기반으로 한 이미지 생성 연구 ([SPADE](https://arxiv.org/abs/1903.07291)), text-to-image 연구 ([Dalle2](https://cdn.openai.com/papers/dall-e-2.pdf)) 등등이 있다. 하지만 이런 컨트롤 방식들은 사용자의 다양한 요구를 만족시키기 어렵고 이상적인 이미지 합성을 컨트롤 하기위해 아래의 3가지 조건이 충족 되어야한다고 정의한다.

1. **유연성(Flexibility)**: 다양한 공간적 속성을 조절할 수 있어야함.
2. **정밀성(Precision)**: 공간적 속성을 높은 정밀도로 조절할 수 있어야함.
3. **일반성(Generality)**: 특정 카테고리에 국한되지 않고 다양한 객체 카테고리에 적용 가능해야함.

이러한 3가지 조건들 달성하기 위해 저자들은 생성모델에서 point-based manipulation이 가능한 방법론을 제안한다. 클릭을 통해 다수의 handle points와 그에 대응되는 target points가 주어졌을 때, handle points를 대응되는 target points로 옮기는 것을 목표로 한다. 관련 선행연구로 [UserControllableLT](https://arxiv.org/abs/2208.12408) 가 존재하지만 2가지 명확한 한계점이 있다. 1) 다수의 handle points에 대해서 잘 동작하지 못함, 2) handle points를 정확하게 target points 에 도달하지 못함. 본 논문에선 위의 3가지 이미지 합성 컨트롤 조건을 충족시키며 기존 연구의 2가지 한계점을 극복하기 위해 2개의 문제를 다룬다.

1. Supervising the handle points to move towards the gargets
2. Tracking the handle points

이 두 문제는 오로지 GAN의 feature space에서 연산되며 handle point를 추적하기위한 추가적인 네트워크가 필요없는 강점이 있으며 몇초밖에 안걸리는 매우 빠른 연산이 가능하다.



# 2. Method

본 연구는 사전 학습 된 GAN 모델에서 클릭을 통해 handle points, target points 를 생성한 후 이미지를 컨트롤 하는 방법을 다룬다. 

* StyleGAN2 모델 기반.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290089333-717c2ae1-a202-481f-a23b-c4e98bfe3f66.png">     
  </figure> 
</div>

## 2.1 Interactive Point-based Manipulation

