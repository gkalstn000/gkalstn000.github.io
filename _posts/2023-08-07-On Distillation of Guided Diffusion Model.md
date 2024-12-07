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
> \mathbb{E}_{t \sim U[0, 1], x \sim p_{data}(x), z_t \sim q(z_t | x)}[\omega(\lambda_t) \,\left\|  \
>  \, \hat{x}_{\theta}(z_t)-x \right\|^2_2] \tag{1}
> $$

* $$\lambda_t = \log[\frac{\alpha^2_t}{\sigma^2_t}]$$ 는 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) 의 signal-to-noise ratio 
* $q(z_t \mid x)= N(z_t; \alpha_tx, \sigma^2_tI)$ 와 $\omega(\lambda_t)$ 는 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) 의 pre-specified weighting function



학습 완료 된 model $\hat{x}_{\theta}$ 은 DDIM 을 통해 sampling 할 수 있음. DDIM sampler 는 $z_1 \sim N(0, I)$ 에서 시작해 아래와 같은 update 방식을 따름.



> $$
> z_s = \alpha_s\hat{x}_{\theta}(x_t) + \sigma_s\frac{z_t - \alpha_t\hat{x}_{\theta}(x_t)}{\sigma_t},\quad s = t - \frac{1}{N} \tag{2}
> $$

$N$ 은 sampling step의 총 횟수.

최종 sample은 $\hat{x}_\theta(z_0)$ 으로부터 생성됨.



## Classifier-free guidance

Classifier-free guidance (CF-guidance)는 class-conditioned diffusion model 의 샘플 품질을 향상시키기 위한 방법으로, 여러 최신 SOTA 클래스 조건부 모델들 ([GLIDE](https://arxiv.org/abs/2112.10741), [Stable Diffusion](https://arxiv.org/abs/2112.10752), [DALL-E 2](https://cdn.openai.com/papers/dall-e-2.pdf))에서 사용되고 있다.

이 방법은 Guidance weight paramete $w\in \mathbb{R}^{\geq 0}$를 사용하여 diffusion model의 샘플 품질과 다양성 사이의 균형을 조절한다.

**학습 시**:

- 클래스 조건 $c$ (예: 클래스 라벨, 텍스트 프롬프트)를 확률적으로 0으로 설정하여 조건부 및 비조건부 디퓨전 모델을 학습.
  - $\hat{x}_\theta (x_t, c)$
  - $$\hat{x}_\theta (x_t, \emptyset)$$

**테스트 시**:

- $$\epsilon_\theta = (1+w)\hat{x}_\theta (x_t, c) - w\hat{x}_\theta (x_t, \emptyset )$$

각 sampling 스텝마다 두 번의 평가가 필요하기 때문에, 이 접근법은 상당한 계산 비용/속도가 요구된다.



## Progressive distillation

[Progressive distillation diffusion](https://arxiv.org/abs/2202.00512)은 본 논문에서 사용된 기본선 (baseline)으로 볼 수 있다. 이 방식은 distill 학습 방법을 통해 teacher 모델이 2단계에서 진행하는 noise prediction을 student 모델이 단 1단계만에 수행할 수 있게 만들며, 이 과정을 반복하여 샘플링 스텝의 횟수를 줄이는 방식을 제안한다.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288609833-2cb67889-dd04-4ba1-9f97-f0c8a2fbab4a.png">
    <figcaption>Progressive distillation algorithm</figcaption>   
  </figure> 
</div>




하지만 progressive distill 연구에서는 CF-guidance 적용이 되지 않아, sample 속도는 빨라졌더라도 sample quality가 떨어지는 한계점이 있다.

## Latent diffusion models (LDMs)

LDMs (Latent Diffusion Models)는 전통적인 diffusion 모델들과 달리 training/inference의 효율성을 높이기 위해 pixel-space가  아닌 latent-space에서 diffusion 계산을 수행한다. 이 모델들은 입력 이미지를 latent-space로 변환하고, 이를 다시 복원할 수 있는 사전 학습된 autoencoder를 사용한다. 이렇게 압축된 상태로 diffusion 과정에 활용되어 계산 효율성을 높인다.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288613664-e53b4fdc-19ef-448c-9ff0-5caf06916589.png">
    <figcaption>LDMs</figcaption>   
  </figure> 
</div>

---

본 논문에서는 제안하는 학습 방식이 pixel-space, latent-space에서도 잘 통한다는 것을 보여준다.

* CF-guidance의 sample 퀄리티를 유지하며
* Progressive distillation 의 빠른 sampling 속도를 가지는

# 3. Distilling a guided diffusion model

CF-guided diffusion model을 distilling 하는 것이 목표.

**Given trained guided model** $\hat{x}_\theta(\cdot, \cdot)$ , 제안하는 학습 방식은 2단계로 이루어져 있다.

### 3.1 Stage-one distillation

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288645151-af9622a5-0fd9-4b34-b8c0-27641d0a03f8.png">
    <figcaption>CF distill</figcaption>   
  </figure> 
</div>

Stage-one distillation의 목표는, 임의의 time-step $t \in [0, 1]$에서 student model $\hat{x}_{\eta_1}(z_t, c, w)$의 output이 teacher model의 output과 동일하게 만드는 것.

CF-guidance의 핵심 요소는 guidance strength parameter $w$ 로 diffusion model의 diversity 와 sample quality 사이의 trade-off 하도록 하는 것이다. 이 때 $w$ 는 user preference 이기 때문에 student model 학습을 위해 guidance strengths 범위 $[w_{min}, w_{max}]$ 를 설정하고 이 $w$ 를 이용해 학습을 진행한다.

> $$
> \mathbb{E}_{w \sim p_w, \, t \sim U[0, 1], \, x \sim p_{data}(x)}[\omega(\lambda_t)\left\| \epsilon_{\eta_1} - \epsilon_\theta \right\|^2_2]
> $$
>
> * $p_w(w) = U[w_{min},w_{max}]$
> * $\epsilon_{\eta_1} = \hat{x}_{\eta_1}(z_t, c, w)$
> * $$\epsilon_{\theta} = (1+w) \cdot \hat{x}_\theta(z_t, c) - w \cdot \hat{x}_\theta(z_t, \emptyset)$$

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288650972-b244ca8f-f0cd-4140-861e-9fa7b3ceef4b.png">
    <figcaption>Stage one algorithm</figcaption>   
  </figure> 
</div>



### 3.2 Stage-two distillation

이번 section 에선 **3.1 Stage-one** 에서 distilled 된 모델 $\hat{x}_{\eta_1}$ 의 sampling step 수를 줄이는 것을 목표로 dilstillation 을 수행한다.

> $$\hat{x}_{\eta_1}$$ 는 teacher model이고, $$\hat{x}_{\eta_1}$$ 이 2-step 을 써야 예측할 수 있는 값을 student model $$\hat{x}_{\eta_2}$$ 는 1-step 만에 예측할 수 있도록, 즉 총 step 수가 절반이 되도록 distillation 을 진행한다.
>
> * 선행연구: [progressive distillation diffusion](https://arxiv.org/abs/2202.00512) 
>   * 선행 연구에서의 distill 모델은 unconditional diffusion model에 초점을 맞춘 결과로, Classifier-Free Guidance (CF-Guidance)를 직접 적용하는 데 어려움이 있었다. 이는 CF-Guidance가 샘플링 시마다 두 번의 추출을 요구하기 때문. 그러나 stage-one에서는 $\hat{x}_{\eta_1}$를 이용하여 CF-Guidance를 단 한 번의 추출로 처리할 수 있도록 $w$를 모델 입력으로 적용하였고, 이 접근법 덕분에 progressive distillation diffusion 학습 방식을 효과적으로 적용할 수 있게 되었음.



$N$ 은 student의 total sampling 횟수 이고 $w \sim U[w_{min}, w_{max}]$ , $$t \sim \left\{ 1, \, \dots, \, N \right\}$$ 이 주어졌을 때 student $\hat{x}_{\eta_2}$ 의 output을 teacher $$\hat{x}_{\eta_1}$$ 의 2-step DDIM output과 같아지도록 학습한다.

* teacher의 첫번 째 sampling step: $\frac{t}{N} \sim t - \frac{1}{2N}$
* teacher의 두번 째 sampling step: $t - \frac{1}{2N} \sim t - \frac{1}{N}$

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288670805-be308c14-5ab9-4e90-b3b7-86c7fc10c727.png">
    <figcaption>Stage-two istillation</figcaption>   
  </figure> 
</div>

Teacher 의 $2N$ step을 student 의 $N$ step으로 distilling 한 뒤, $N$ step student는 새로운 teacher가 되어 새로운 $\frac{N}{2}$ step 의 student 학습을 반복한다.

매 step마다 student model의 parameter는 teacher의 parameter로 초기화 된다.



### 3.3 $N$-step deterministic and stochastic sampling

Distillation을 통해 teacher가 수행해야 할 sampling step보다 절반이 줄어든 $\hat{x}_{\eta_2}$ 는 특정 구간의 guidance 범위에 $$w \in \left [ w_{min}, w_{max} \right ]$$ 대해 DDIM 을 통해 원하는 이미지 sampling이 가능하다. 

이 때 DDIM 은 *deterministic*  sampling procedur 이지만 $N-step$ *stochastic* sampling 을 활용한 distillation 훈련도 가능하다고 한다.

* 아마도 본 논문에선 DDIM을 예시로 썼지만, 실험을 해본 결과 다른 sampling 방식또한  distillation 가능하다 라는 것을 보여주고 싶었는듯..

> Stage-one, Stage-two 에서 distillation 할 때 deterministic sampling 방식인 DDIM을 활용했으나 stochastic sampling 방식을 통해서도 distillation 학습 및 sampling이 가능하다라는 것을 설명하는 챕터인듯하다.
>
> Stochastic sampling 관련 선행연구로는 아래의 대표적인 논문이 있는 것같은데 ODE와 score based 개념이 필요한 너무 어려운 논문이라 지금 당장은 이 챕터를 이해하기엔 무리가 있을 듯하다.
>
> * [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf) 
> * [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288965401-d44f7f9b-0c04-4e94-bb72-684543a01ced.png">
    <figcaption>Stochastic sampling</figcaption>   
  </figure> 
</div>

원래 step보다 2배가 긴 deterministic sampling step을 한번 거친 뒤, stochastic step을 뒤로 적용하며 sampling 을 수행한다고 한다.



# 4. Experiments

저자들이 실험을 통해 보여주고자 하는 주요 내용은, 제안된 2-stage distillation 학습 방법이 pixel-space diffusion과 latent-space diffusion 모두에 적용 가능하며, 더 나아가 text-guided image editing, inpainting 등 다양한 diffusion task에도 효과적으로 사용될 수 있다는 것이다.

실험 결과에 따르면, 이 방법은 단 2~4 step 만으로도 경쟁력 있는 성능(competitive performance)을 보여준다. 이는 훨씬 적은 샘플링 스텝으로도 높은 품질의 결과를 얻을 수 있음을 의미하며, 이는 디퓨전 모델의 효율성과 범용성을 대폭 향상시키는 중요한 발전이다.

## 4.1 Distillation for pixel-space guided model

제안하는 학습 방식이 pixel-space에서 통하는지 입증하는 실험

**Setting**

* Dataset: ImageNet 64x64, CIFAR-10
* $$ \left [ w_{min}, w_{max} \right ] = \left [ 0, 4 \right ]$$
* 비교 대상
  * Teacher full step (DDPM == DDIM 1024 steps)
  * Teacher DDIM samplings
  * Single-w
    * distilled 된 모델  $\hat{x}_{\eta_2}$ 의 다양한 $w$ 값을 실험하는 것 외에도, 특정 고정된  $w$값으로 모델을 distill 시켜 그 결과를 기준점으로 사용하여,  $w$의 영향을 더 명확히 이해하기 위해 세팅.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288967003-4a5971eb-7c62-4b48-9207-c8f9a4ccd599.png">
    <figcaption>Pixel space comparison. D/S는 각각 deterministic/stochastic sampling 을 의미</figcaption>   
  </figure> 
</div>

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288969328-e0eee082-598b-426d-9e77-19002332cb3b.png">
    <figcaption>guidance weight 변화에 따른 FID/IS 변화</figcaption>   
  </figure> 
</div>

최초의 teacher model의 1024 sampling step과 student의 4~16 sampling 된 결과물의 score가 거의 비슷하거나 더 좋은 케이스를 확인할 수 있음.

고정된 $w$ 로 distill 학습 된 모델과 비교해도 competive 한 성능을 보여줌.



## 4.2 Distillation for latent-space guided models

뒷 내용들은 딱히 분석할 만한 실험 결과가 없어서 생략.

Latent-space에서도 잘 동작하며, 다양한  task에도 문제 없다는 내용.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288969977-a952df3b-71f9-4fd0-8594-83771a34c51c.png">
    <figcaption>Text-to-Image</figcaption>   
  </figure> 
</div>

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/288970024-c2526fa2-614b-4118-9625-6d037b0c9f59.png">
    <figcaption>Text guided Image-to-Image translation</figcaption>   
  </figure> 
</div>
