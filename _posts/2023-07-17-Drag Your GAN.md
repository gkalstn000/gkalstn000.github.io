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

>*DragGAN* allows users to control various spatial attributes by **'dragging'** the content of any GAN-generated images.
>
>In real-world applications, a critical functionality requirement of image synthesis methods is controllability.
>
>Ideal controllable image synthesis approach properties
>
>* **Flexibility**: control different spatial attributes
>* **Precision**: control the spatial attributes with high precision
>* **Generality**: applicable to different object categories but not limited to a certain category

<div style="text-align: center;">   
  <figure>     
    <img src="https://github.com/XingangPan/DragGAN/raw/main/DragGAN.gif">     
  </figure> 
</div>

**Web Demos**

[![Open in OpenXLab](https://camo.githubusercontent.com/a28f1d6dc75b31d084cb14e4c6f5bbd0f97bfbecd7bf260fac8dbd95df1c9430/68747470733a2f2f63646e2d7374617469632e6f70656e786c61622e6f72672e636e2f6170702d63656e7465722f6f70656e786c61625f6170702e737667)](https://openxlab.org.cn/apps/detail/XingangPan/DragGAN)

[![Huggingface](https://camo.githubusercontent.com/3cc55bf2ba0bc623f8f32e700b9834d27767be50da0a720d5a7f083768be244d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d4472616747414e2d6f72616e6765)](https://huggingface.co/spaces/radames/DragGan)



# Introduction

2014년 GAN(Generative Adversarial Networks)의 등장과 2020년 DDPM(Denoising Diffusion Probabilistic Models)의 등장은 이미지 생성 연구에 큰 변화를 가져왔다. 이러한 발전은 이미지 생성 분야에서 놀라운 성과를 가져왔지만, 실제 응용에서는 단순히 조건에 맞는 이미지를 생성하는 것뿐만 아니라 생성된 이미지를 세밀하게 조절할 수 있는 능력이 중요하다.

이미지 조절에는 인물의 위치, 감정, 표정 변경과 같은 다양한 요소들이 포함됩니다. 사용자의 이러한 다양한 요구를 만족시키기 위해서는 다음 세 가지 요소가 필요하다:

1. **유연성(Flexibility)**: 다양한 공간적 속성을 조절할 수 있어야함.
2. **정밀성(Precision)**: 공간적 속성을 높은 정밀도로 조절할 수 있어야함.
3. **일반성(Generality)**: 특정 카테고리에 국한되지 않고 다양한 객체 카테고리에 적용 가능해야함.

본 논문에서는 사용자가 변형하고 싶은 부위에 대한 핸들 포인트를 클릭하여 형성하고, 변경하고자 하는 방향에 대한 타겟 핸들 포인트를 설정함으로써, 원하는 방식으로 이미지를 조절할 수 있는 기능을 소개한다.

