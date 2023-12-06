---
layout: post
title:  "On Distillation of Guided Diffusion Models"
categories: 논문리뷰
date: 2023-08-07 11:40:18 +0900
tags: Diffusion 생성모델 Distillation
mathjax: True
author: Haribo
---
* content
{:toc}
**Full Citation**: "Meng, Chenlin, et al. "On distillation of guided diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023."\
**Link to Paper**: [https://arxiv.org/pdf/2210.03142.pdf](https://arxiv.org/pdf/2210.03142.pdf) \
**Conference Details**: CVPR 2023 (우수논문)

---

> * 느린 샘플링이라는 기존 diffusion 모델의 한계를 효과적으로 해결한 [연구](https://arxiv.org/abs/2202.00512)는 있으나, 이는 unconditional diffusion model에 한정된다.
>
> * High-Resolution conditional image generation 을 위해서는 [**Classifier-free guidance**](https://arxiv.org/abs/2207.12598)가 필요하지만, 이 방법은 많은 연산을 필요로 하며 샘플링 속도가 느리다.
>
> * 이 논문은 Classifier-free guidance를 사용하면서도 빠른 샘플링을 가능하게 하는 **새로운 학습 기법**을 제안하며, 이 기법은 단 1~4 단계만으로도 기존 모델들과 비교할 수 있는 성능을 보여준다.
>   * Pixel-space diffusion: ImageNet 64x64, CIFAR-10 데이터셋에서 단 4 denoising step만에 기존 모델과 비슷한 성능.
>
>   * Latent-space diffusion: LAION  데이터셋에서 1~4 denoising step만에 기존 모델과 비슷한 성능.
>
>   * Text-to-Image diffusion: 2~4 denoising step만에 기존 모델과 비슷한 성능
>



<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288264103-46f345e8-3cac-4bc9-9649-81984f001338.png">     
  </figure> 
</div>












# 1. Introduction

Denoising Diffusion Probabilistic Models (DDPMs)는 이미지 생성, 오디오 합성 등 다양한 분야에서 뛰어난 성과를 보여주었다. 특히 [Classifier-free guidance](https://arxiv.org/abs/2207.12598)를 통해 생성 품질이 크게 향상되었음.

* Classifier-free guidance를 사용한 주요 연구들로는 [GLIDE](https://arxiv.org/abs/2112.10741), [Stable Diffusion](https://arxiv.org/abs/2112.10752), [DALL-E 2](https://cdn.openai.com/papers/dall-e-2.pdf) 등이 있다.

하지만 Classifier-free guidance는 이미 긴 diffusion sampling 시간을 더욱 연장시켜 실제 어플리케이션에 적용하는 데 한계를 가지고 있었다. 이 문제를 해결하기 위해, [progressive distillation diffusion](https://arxiv.org/abs/2202.00512) 학습 방식이 제안되었지만, 이는 classifier-free guided diffusion model에 바로 적용할 수 없는 한계가 있었다.

이에 저자들은 two-stage distillation approach를 제안하여 classifier-free guidance의 품질은 유지하면서도 샘플링 효율을 향상시키는 방법을 개발했다. 이 새로운 방식을 통해 학습된 모델들은 다양한 디퓨전 모델에서 적은 샘플링 횟수로도 기존 모델들과 비슷한 성능을 보여준다.

* Pixel-space diffusion model: 4 denoising step 만으로 시각적으로 충분한 고퀄리티 이미지, 4~16 step 만에 comparable 한 FID, IS score
* Latent-space diffusion model: 1~4 denoising step 만으로 시각적으로 충분한 고퀄리티 이미지, 2~4 step 만에 comparable 한 FID score.
* Text-to-Image & image in-painting diffusion model: 2~4 step 만에 comparable한 시각적으로 충분한 고퀄리티 이미지



### 제안하는 two-stage 학습 방식

First stage: Single student model to match combined output of the two diffusion models of the teacher.

Second stage: Progressively distill the model learned from the first stage to a fewer-step model using [progressive distillation diffusion](https://arxiv.org/abs/2202.00512) 



# 2. Backgrouond on diffusion diffusion models

* Data distribution $p_{data}(x)$ 로 부터 sample 된 $x \sim   p_{data}(x)$
* Noise scheduling function. $\alpha_t$ 와 $\sigma_t$
* 학습 할 diffusion model $\hat{x}_{\theta}$



Weighted mean squared error.  



> $$
> \mathbb{E}_{t \sim U[0, 1], x \sim p_{data}(x), z_t \sim q(z_t | x)}[\omega(\lambda_t)\left\| \hat{x}_{\theta}(z_t)-x \right\|^2_2]
> $$

* $$\lambda_t = \log[\frac{\alpha^2_t}{\sigma^2_t}]$$ 는 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) 의 signal-to-noise ratio 
* $q(z_t \mid x)= N(z_t; \alpha_tx, \sigma^2_tI)$ 와 $\omega(\lambda_t)$ 는 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) 의 pre-specified weighting function



학습 완료 된 model $\hat{x}_{\theta}$ 은 DDIM 을 통해 sampling 할 수 있음. DDIM sampler 는 $z_1 \sim N(0, I)$ 에서 시작해 아래와 같은 update 방식을 따름.



> $$
> z_s = \alpha_s\hat{x}_{\theta}(x_t) + \sigma_s\frac{z_t - \alpha_t\hat{x}_{\theta}(x_t)}{\sigma_t},\quad s = t - \frac{1}{N}
> $$

$N$ 은 sampling step의 총 횟수.

최종 sample은 $\hat{x}_\theta(z_0)$ 으로부터 생성됨.



## Classifier-free guidance

## Progressive distillation

## Latent diffusion models (LDMs)



# 3. Distilling a guided diffusion model
