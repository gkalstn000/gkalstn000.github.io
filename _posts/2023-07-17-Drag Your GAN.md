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

latent code $w$ 와 GAN을 통해 생성 된 이미지 $$I \in \mathbb{R} ^ {3 \times H \times W}$$ 가 주어지고, user의 입력값이 각각 아래와 같이 주어졌을 때,

* handle points: $$\left\{ p_i = (x_{p,i}, y_{p,i}) \mid i=1,2, \dots, n) \right\}$$
* Target points: $$\left\{ t_i = (x_{t,i}, y_{t,i}) \mid i=1,2, \dots, n) \right\}$$

이미지 내부의 object를 움직이는 것을 목표로 한다. 추가적으로 유저의 선택으로 움직일 부분을 제한하는 binary mask $M$ 도 사용 가능하다.

> $w \in W$ 를 통해 생성된 GAN의 중간 feature map 의 좌표 $p_i=(x_{p,i}, y_{p,i})$ 에 대응되는 벡터값이 $v_i$ 라 가정해보자. 이때, 다른 좌표 $t_i=(x_{t,i}, y_{t,i})$에서 $v_i$와 동일한 벡터 값을 가진 중간 feature map 을 생성하는 $w' \in W$가 존재할 것이고, 본 논문은 이러한 **최적의 $w'$ 를 찾는 문제**로 볼 수 있다.

위 그림에서 확인할 수 있듯이 optimization step은 2개의 sub-step으로 구성되고 서서히 이미지를 변경해가는 과정을 거친다.

1. Motion supervision: latent code $w$ 를 변경해가며 $p_i$ 위치의 벡터값들을 $t_i$ 에 위치하게끔 한다.
2. Point tracking: 변형 된 $w'$를 통해 생성된 조금 변형 된 새로운 이미지 $I'$ 에서,  기존 $p_i$ 위치의 벡터값과 동일한 벡터값을 가지는 위치를 tracking.

이런 최적화 과정을 서서히 거치면서 이미지 내의 객체를 움직이게끔 하고 보통 30~200 step이 소요된다고 한다.



## 2.2 Motion Supervision

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290369187-1e92d444-e6b3-44a7-a558-7637d88ad447.png">     
  </figure> 
</div>

이 연구에서는 GAN이 생성한 이미지에서 point motion supervision 을 위한 새로운 모션 감독 손실 함수(motion supervision loss)를 제안한다. 이 손실 함수는 추가적인 네트워크 없이도 작동하며, $W$ 내의 latnet code $w$를 통해 생성된 중간 feature map $F$ 에서 좌표 $p_i$의 벡터 값 $v_i$가 다른 좌표 $t_i$에서 관찰되는 새로운 중간 feature map $F'$ 를 생성하는 $w' \in W$를 찾는데 목적을 둔다.



**Notations**

* $F$ :StyleGAN2의 6번째 layer의 feature map을 원본 이미지 $I \in \mathbb{R} ^ {3 \times H \times W}$ 의 사이즈 $(H, W)$로 resize한 tensor.
*  $p_i$ : Handle point
* $t_i$ : Target point
* $\Omega_{1}(p_i, r_1)$ : $p_i$ 주변 $r_1$ 반경의 좌표들의 집합.

> $$
> L = \sum_{i=0}^{n}\sum_{q_i \in \Omega_{1}(p_i, r_1)} \left\| F(q_i) - F(q_i + d_i) + \right\| + \lambda \left\| (F-F_0) \cdot (1-M) \right\|
> $$
>
> * $F(q)$: feature map $F$ 의 $q$ 좌표에 해당하는 vector 값
> * $d_i = \frac{t_i-p_i}{\left\| t_i-p_i \right\|_2}$  는 $p_i$ 에서 $t_i$ 방향의 normalizaed vector값이다.
> * $F_0$ 는 초기 $w$로 만든 feature map을 의미함.
> * $M$: binary mask로 unmasked된 위치는 변경하지 않도록 제약.
>
> Back-propagation 을 통해 $w$를 업데이트 할 때 $F(q_i)$ 는 freeze 하고, $F(q_i+d_i)$를 통해서만 진행.
>
> * $p_i$ 값만 $p_i + d_i$ (handle point를 target point 방향) 로 움직이게끔 하기 위해서.



## 2.3 Point Tracking



<div style="text-align: center;">   
  <figure>     
    <img src="https://camo.githubusercontent.com/2eedfeb3fce75496f61186223db11db77a508fe74c5e8b71f70591cd16c6528a/68747470733a2f2f7061727469636c652d766964656f2d7265766973697465642e6769746875622e696f2f696d616765732f70757070795f776964652e676966">     
  </figure> 
  <figcaption>Point tracking 예시 (PIPs, ECCV2022)</figcaption>   
</div>

Motion supervision을 통해 업데이트된 새로운 latent code $w'$로부터 생성된 새로운 feature map $F'$와 이미지 $I'$를 얻을 수 있다. 이 과정에서, $F(q_i)$를 $(q_i + d_i)$ 위치에 위치시키기 위해 $w'$를 생성하는 것이 목표이지만, 실제로 $(q_i + d_i)$ 위치에 $F(q_i)$가 있는지는 알 수 없다. 기존 연구에서는 이러한 point $F(q_i)$의 추적을 위해 별도의 AI 모델을 사용했지만(상기 이미지 예시 참조), 본 논문에서는 추가 학습 없이 중간 feature map의 discriminative한 특성을 활용하여 point tracking을 수행한다.

**Notations**

* New latent code $w'$, new feature map $F'$, and a new image $I'$
* Feature map of initial handle point $f_i = F_0(p_i)$
* Patch around $p_i$ as $$\Omega_2(p_i, r_2) = \left\{ (x, y) \mid \;\;  \mid x - x_{p,i} \mid < r_2 , \, \;\;  \mid y - y_{p,i}\mid < r_2  \right\}$$
* $p_i := \underset{q_i \in \Omega_w(p_i, r_2)}{\mathbf{argmin}} \left\| F'(q_i) - f_i \right\|_1$

새로 생성 된 $F'$ 에서 $p_i$ 를 기준으로 주변 패치들 중 최초의 feature map $F_0$ (이전 step feature map 아님) 의 $p_i$ 좌표 vector $f_i$ 와 차이가 가장 적은 위치 찾기.



# 3. Experiments

실험에서 주요하게 확인해봐야할 내용들은 아래와 같다.

1. Qualitative, Quantative comparison
2. Point tracking 할 때 선행연구 모델 안쓰고 단순히 feature map끼리 계산했는데 효과가 더 좋을까?
3. 다수의 points들도 효과적으로 컨트롤가능할까?



**실험 세팅**

> * Backbone generator: Pretrained StyleGAN2
> * Dataset
>   * StyleGAN2 ep이터셋과 동일 (FFHQ-얼굴, AFHQC-얼굴, LSUN-차/고양이 등등 여러 도메인)
> * Baseline
>   * UserControllableLT [*Computer Graphics Forum*. 2022]
>     * Transformer를 활용한 controlable GAN 모델, 문제 setting이 DragGAN과 유사.
>     * Mask input이 허용되지 않음
> * Point tracking 모델
>   * RAFT[ECCV 2020], PIPs[ECCV 2022]
>   * 별도의 Neural Network를 학습시켜 point tracking.

## 3.1 Qualitative Evaluation

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290423471-d3626bc3-678a-4330-92a5-78c37cde4204.png">     
  </figure> 
</div>

UserControllableLT cannot faithfully move the handle points to the targets and often leads to undesired changes in the images

* E.g., the clothes of the human and background of car.

## 3.2 Qialitative tracking comparison

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290424692-5fa27d2a-68ed-4a80-b4db-92fbe1c6ad51.png">     
  </figure> 
</div>

Point tracking시 선행연구 (PIPs, RAFT) 인공지능 모델은 제대로 된 tracking이 안되는 모습을 볼 수 있음. 특히 point tracking을 하지 않았을 경우 handle point에 제대로 된 update가 진행되지 않는다.

## 3.3 Face landmark manipulation

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290425522-84bd2a72-622b-45fc-b4e4-80c45e1796f8.png">     
  </figure> 
  <figcaption>2, 3, 4열의 숫자는 MD score</figcaption>   
</div>

**실험 세팅**

1. Randomly generate two face images using the StyleGAN [CVPR 2019] trained on FFHQ and detect their landmarks.
   * Face landmark detect tool: *Off-the-shelf* [King, 2009]

2. Manipulate the landmarks of the first image to match the landmarks of the second image.

3. After manipulation, we detect the landmarks of the final image and compute the mean distance (MD) to the target landmarks.



MD (Mean Distance) score는 landmarks를 target poisition 으로 얼마나 잘 옮겼는지를 보여주는 지표.

남의 얼굴 랜드마크로 변형하는 실험인데 handle point가 많아도 잘 동작하는 것을 확인할 수 있음.

## 3.4 Paired image reconstruction

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290426855-d49de783-74fd-4dfd-a76d-706f659c4bb1.png">     
  </figure> 
  <figcaption>UserControllableLT [Computer Graphhics Forum. 2022]</figcaption>   
</div>

**실험 세팅**

1. Sample a latent code $w_1$ and randomly perturb it to get $w_2$.
   * Let $I_1$ and $I_2$ be the StyleGAN images generated from the two latent codes.

2. Compute optical flow between $I_1$ and $I_2$ and randomly sample 32 pixels from the flow field as the user input.
3. Goal is reconstruct $I_2$ from $I_1$



다양한 domain의 데이터셋에서 baseline인 UserControllableLT, 선행연구 point tracking ablation 그리고 제안하는 모델을 비교하는 실험을 보여준다.

그나마 PIPs의 point tracking이 괜찮은 성능을 보여주지만 **3.3 Face landmark manipulation** 에서 확인할 수 있듯 속도면에서 제안하는 방식이 훨씬 빠르다.



# Discussion 

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/290428221-a2360527-1b29-4081-a426-b88f363cdcb5.png">     
  </figure> 
</div>

* Effects of mask

  * DragGAN은 binary mask를 이용해 원치않는 부분은 변경이 안되도록 할 수 있음.

  * Point-based manipulation often has multiple possible solutions and the GAN will tend to find the closest solution in the image manifold learned from the training data.

* Limitations

  * Editing quality is affected by the diversity of training data.
    * 그림 (a)를 보면 해당 데이터셋은 추로 차렷자세가 많기 때문에 손,발의 각도를 바꿔도 강제로 차렷 손목 발목이 유지됨.

  * Handle points in texture-less regions sometimes suffer from more drift in tracking
    * Motion supervision 특성상 비슷한 handle과 target이 비슷한 특징을 가진 부위면 차이점 계산이 어려워 제대로 변형이 안된다.

* Social impacts
  * Could be misused to create fake pose, expression, shape.

