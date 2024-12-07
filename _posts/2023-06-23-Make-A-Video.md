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

>* 선행 Text-to-Video 연구들은 다수의 video-text pair 데이터셋이 필요했으나, 사전학습 된 Diffusion 모델의 능력을 활용해 video-text 데이터셋 없이 video 데이터셋만을 활용해 고퀄리티 text-to-video 생성모델 학습 방식을 선보임. 
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



# 2. Method

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293015208-39252387-9cbc-478d-b8ea-846a436d1202.png">     
  </figure> 
  <figcaption>Make-A-Video high-level architecture</figcaption>   
</div>



Make-A-Video는 3개의 main component로 구성되어 있음.

1. Base T2I model trained on text-image pairs
2. Spatiotemporal Convolution and Attention layers
   * 일반 Diffusion 네트워크에 temporal dimension(연산 계층)을 확장한 연산 block
3. Frame interpolation network

Make-A-Video의 수식은 아래와 같이 정리된다:

> $$
> \hat{y}_t = SR_{h} \circ SR^{t}_{l} \circ \uparrow _{F}\circ D^{t} \circ P \circ (\hat{x}, C_{x}(x))
> $$
>
> * $\hat{y}_t$ : Generated Video
> * $SR_{h}, SR^{t}_{l}$: Spatial and spatiotemporal super-resolution networks
> * $\uparrow _{F}$: Frame interpolation network
> * $D^t$ spatiotemporal decoder
> * $P$: Prior
>   * Dalle2 와 비슷하게 $\hat{x}$ 와 $C_x(x)$ 를 condition으로 받아 image embed를 생성하는 일반적인 Diffusion model.
>   * 여기에 temporal 연산이 가능하도록 내부의 conv, attn block을 확장.
> * $\hat{x}$ : BPE-encoded text
> * $C_x$: CLIP text encoder
> * $x$: input text 



## 2.1 Text-to-Image Model

> Video를 학습하기 전, text를 받아 image를 생성할 수 있도록 학습을 시키는 과정.
>
> Image 생성 학습이 끝나면 그 능력을 그대로 video 생성쪽으로 전이 시키는 학습을 거침.



Prior $P$ (일반적인 Diffusion 모델)에 temporal component를 확장 시키기 전 text-image pair 데이터셋으로 학습 시킴.

* Dalle2 방식 [참조](https://arxiv.org/abs/2204.06125)

텍스트로부터 high-resolution 이미지를 생성하기 위해 아래와 같은 과정을 사용

1. Prior network $P$ 는 text embeddings $x_e$와 BPE encoded text tokens $\hat{x}$ 를 입력받아 image embeddings $y_e$ 를 생성
   * $y_e = P(\hat{x}, x_e)$
2. Decoder $D$ 는 $y_e$ 를 condition으로 low-resolution $64 \times 64$ RGB 이미지 $\hat{y}_l$ 을 생성.
   * $\hat{y}_l = D(y_e)$
3. Two super-resolution networks $SR_{h}, SR^{t}_{l}$ 는 $\hat{y}_l$ 의 해상도를 $256 \times 256$ 그리고 $768 \times 768$ 로 확장시켜 최종적으로 이미지 $\hat{y}$ 을 생성.
   * $\hat{y} = SR_{h} \circ SR^{t}_{l} (\hat{y}_l)$



### 2.2 SpatioTemporal Layers



<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293116973-1ebde152-0cea-482d-a67e-200168b1d4dd.png">     
  </figure> 
  <figcaption>(왼)Pseudo-3D Conv (오)Pseudo-3D Attn</figcaption>   
</div>



Conv, Attn block들로 이루어진 UNet 구조의 Diffusion 모델에 temporal dimension을 확장하기 위해 Conv와 Attn에 modify를 해야한다.

* fully-connect같은 기타 layer는 딱히 시간의 영향을 안받으므로 수정 안함.



Temporal modification이 적용되는 모델들은 아래와 같음

* Decoder $D^t$
* Frame-interpolation $\uparrow _{F}$
* Super-resolution networks $SR^t_l$



이 때 $768 \times 768$ 해상도로 super-resolution 하는 $SR_h$ 의 경우 attention 메모리 부족 문제로 temporal 을 고려한 고해상도 확장은 적용되지 않음.



### 2.2.1 Pseudo-3D Convolution Layers

이미지와는 다르게 4D 사이즈 $(C, F, H, W)$를 가진 동영상을 처리하려면 3D-Conv 연산이 필요한데, 이는 상당한 연산량과 메모리를 요구한다. 이에 대한 해결책으로, [separable convolution](https://arxiv.org/abs/1610.02357)에서 영감을 받아 기존의 2D Conv 뒤에 1D Conv를 추가함으로써 temporal 연산을 포함한 Pseudo-3D Conv를 제안한다.

게다가 기존 2D Conv에 1D Conv를 추가한 방식이라 pre-trained 된 (Sec.2.1) 기존 네트워크의 가중치를 그대로 활용할 수 있음.

* 1D conv만 새롭게 init 하면 됨.



```python
    def forward(self, x, enable_time = True):
        b, c, *_, h, w = x.shape # x 는 video or image

        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x) # spatial_conv는 일반적인 2D conv

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

        if not enable_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        x = self.temporal_conv(x) # temporal_conv 는 일반적인 1D conv

        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

        return x
```



> $Conv_{P3D} = Conv_{1D}(Conv_{2D}(h) \circ T ) \circ T$



이런식으로 공간적으로 conv 연산을 한번 해준 뒤, $(B, H, W)$ 를 batch 로 묶어 1D Conv 연산을 해주면 Frame간 연산 (시간적 연산) 이 가능하다. 



### 2.2.2 Pseudo-3D Attention Layers

**Pseudo-3D Conv** 와 비슷하게 Attn 연산 또한 reshape를 통해 공간적/시간적 연산을 따로 진행한다. 이 때도 마찬가지로 시간적 연산을 진행할 때 새로 추가된 연산 block에 대해서만 init을 하기 때문에 기존 공간적 attention 가중치는 그대로 활용가능.

```python
 def forward(self, x, enable_time = True):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w) if exists(self.spatial_rel_pos_bias) else None

        x = self.spatial_attn(x, rel_pos_bias = space_rel_pos_bias) + x

        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b = b, h = h, w = w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)

        if enable_time:

            x = rearrange(x, 'b c f h w -> (b h w) f c')

            time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1]) if exists(self.temporal_rel_pos_bias) else None

            x = self.temporal_attn(x, rel_pos_bias = time_rel_pos_bias) + x

            x = rearrange(x, '(b h w) f c -> b c f h w', w = w, h = h)

        if self.has_feed_forward:
            x = self.ff(x, enable_time = enable_time) + x

        return x
```



* 공간적 attention: $(B, F) \; (H, W) \; C$ 에 대해 attention 연산.
* 시간적 attention: $(B,H,W) \; F \; C$  에 대해 attention 연산.



**Frame rate conditioning.**

[CogVideo](https://arxiv.org/abs/2205.15868) 와 비슷하게 Frame condition parameter $fps$ 를 적용.

* $fps$ : number of frames-per-second in generated video.

다양한 frames-per-second를 조건으로 학습하면 inference시 추가적인 control이 가능하다.

**Objectives**

[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)의 $hybrid$ loss 적용. (아래 수식에선 VLB 생략)

> $$L_{decoder} = \mathbb{E}_{C_y(y_0), \epsilon, fps, t} \left [ \left\| \epsilon_t - \epsilon_\theta (z_t, C_y(y_0), fps, t) \right\|^2_s \right ]$$
>
> * $y$ : 입력 비디오
> * $y_0$ : 입력 비디오의 첫 frame (이미지)
> * $C_y(y_0)$ : $y_0$ 에 대한 CLIP image embedding
> * $z_t$ : $y$ 에 $t$  시점의 noise 를 더한 input



## 2.3 Frame Interpolation Network

Sec 2.2 에서 다룬 시공간 확장 외에 생성 된 비디오의 frame을 늘리는 새로운 네트워크 $\uparrow _{F}$ 를 학습한다.

* Frame interpolation: 더 부드러운 비디오 생성
* Frame extrapolation: 기존 비디오 전/후 프레임 생성

$\uparrow _{F}$ 는 spatiotemporal decoder $D^t$ 를 masked frame interpolation task로 finetuning 해서 만든다.

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293145087-ce6c2476-c46d-4ba0-9e2f-1c8d0cc3e1ca.png">     
  </figure> 
  <figcaption>Frame interpolation 설명파트 논문이 상당히 불친절하고 생략이 많아서 대충 그려봄.</figcaption>   
</div>

$fps$ token으로 masked 된 원본 frame을 reconstruction (BERT와 비슷) 하는 과정을 여러번 거쳐 샘플링 된 16 frame의 비디오에서 원본 길이의 frame 비디오를 생성 해냄.

* 실험에서는 76 프레임이 원본 비디오 길이로 설정.



실험에서 $\uparrow _{F}$  는 5 frame 씩 건너뛰며 $D^t$ 에 들어갔다가 $\uparrow _{F}$  에서 76개의 frame으로 생성됨.

* $76 = ((16-1) \times 5 + 1 )$

## 2.4 Training

Make-A-Video의 각 component 들은 독립적으로 각각 학습이 됨.

* Prior $P$ : text를 입력으로 받는 diffusion 모델로 비디오에 대해서는 학습 시키지 않음
* $D^t$ 와 $SR$ 2개 : 처음 이미지로 학습하고 그 다음 비디오에 대해 학습.

각 component가 학습 되고 나면 새로운 temporal layers를 초기화 후 unlabeled video에 대해 fintuning 실시.

원본 비디오에서 16개의 frame들이 random하게 샘플링 되고 각 frame 들은 노이즈가 칠해져서 (diffusion beta function) $D^t$에 들어간 뒤 $\uparrow _{F}$ 로 frame 수 늘려줌.



# 3. Experiments

## 3.1 Datasets and Settings

**Training**

* T2I
  * LAION 2.3B dataset (Text-Image pair)
  * Prior $P$, Decoder $D^t$, Super-resolution $SR^t_l$, $SR_h$ 
* T2V
  * WebVid-10M (Video dataset)
    * Decoder $D^t$, Frame interpolation $\uparrow_F$
  *  HD-VILA-10M (Video dataset)
    * Super-resolution $SR^t_l$

선행 T2V 연구들인 [CogVideo](https://arxiv.org/abs/2205.15868), [VDM](https://arxiv.org/abs/2204.03458) 는 비공개 video-text pair 데이터셋으로 학습 진행했으나, Make-A-Video는 open dataset으로 학습 진행.

**Evaluation**

학습 때 사용되지 않은 template sentence를 선정.

* [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)
  * Action recognition 데이터셋으로 각 클래스마다 한 문장씩 평가용 sentence 로 세팅
  * Eval metric
    * Frechet Video Distance (FVD)
    * Inception Score (IS)
* [MSR-VTT](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf)
  * Microsoft Research Video to Text 데이터 셋으로 영상 하나에 다양한 문장들이 존재함.
  * Eval metric
    * Frechet Inception Distance (FID)
    * CLIPSIM (average CLIP similarity between text and video frames)
* Amazon Mechanical Turk (AMT)
  * Human evaluation 으로 300개의 prompts를 구성
  * 5개의 카테고리 (동물, 판타지, 인물 등등)에 대해 평가진행.
* DrawBench prompts from Imagen (Video gen에 많이 사용되는 벤치마크 prompt)
  * Human evaluation으로 video quality와 text faithfulness를 각각 평가.

## 3.2 Quantitative Results

**Automatic Evaluation on MSR-VTT**

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293313549-69f77fc3-ec3d-4b58-9908-671839137df0.png">     
  </figure> 
</div>

* CogVideo와 Make-A-Video의 경우 각 prompt에 대해 zero-shot 세팅을 진행
* 선행 연구들은 모두 매우 짧은 16 frame 동영상만 생성 가능하기에 Make-A-Video 또한 $16 \times 256 \times 256$ 사이즈의 동영상만 생성.
* Make-A-Video는 zero-shot임에도 불구하고 MSR-VTT에서 학습 된 GODIVA 와  NUWA보다 좋은 성능을 보임.
* 또한 SOTA인 CogVideo 보다 더 좋은 일반화 성능을 보여줌.

**Automatic Evaluation on UCF-101**

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293314534-cd59d501-ee34-4464-80b2-1938f9c92991.png">     
  </figure> 
</div>

UCF-101은 비디오 생성 분야에서 유명한 벤치마크로 최근 T2V 모델 평가에 많이 사용됨. 

* CogVideo UCF-101 데이터셋에 fine-tuning 후 class-conditional 비디오 생성 된 상태.
* VDM은 unconditional 비디오 생성이지만 UCF-101 데이터셋으로 학습 된 상태.
* Make-A-Video는 zero-shot 및 fine-tuning 세팅에 대해 평가 진행
* zero-shot 성능자체가 이미 UCF-101 데이터셋으로 학습 된 다른 연구들의 성능을 거의 뛰어넘음.
* Fine-tuning 세팅에선 SOTA 달성.

**Human Evaluation**

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/293315507-266ee28e-2651-46eb-bfef-83784f841ce7.png">     
  </figure> 
</div>

* 공개된 zero-shot T2V 모델인 CogVideo 와 비교
* DrawBench 와 자체 테스스세트에서 평가.
* [VDM 웹페이지](https://video-diffusion.github.io/) 에 게시된 28개의 비디오에 대해 추가로 평가.
  * VDM에 유리함.
* 각 prompt에 대해 무작위 8개의 비디오 생성 후 8회에 걸쳐 평가결과의 평균치 기록.
  * $76 \times 256 \times 256$ 해상도 비디오 생성.
* Make-A-Video는 비디오 품질 및 Text-Video faithfulness에 대해 모든 벤치마크와 비교에서 더 나은 성능.



**Frame interpolation 평가**

실험 결과는 없고 말로만 쓰여있음.

* FILM과 비교하여 자체 frame interpolation 네트워크를 평가.
* DrawBench와 자체 평가 세트에서 텍스트 프롬프트를 기반으로 1 FPS의 저 프레임 비디오를 생성한 후, 각 방법을 사용하여 4 FPS로 업샘플링.
* 평가자들은 자체 평가 세트에서 62%, DrawBench에서 54%의 비율로 더 현실적인 움직임을 보여주는 자체 방법을 선택.
* Make-A-Video는 객체가 현실세계에서 가지는 움직임에 대한 지식이 훨씬 더 뛰어나다는 것을 나타냄.



## 3.3 Qualitative Results

다양한 실험 결과 프로젝트 블로그 [주소](https://makeavideo.studio/)

<div style="text-align: center;">   
  <figure>     
    <img src="https://makeavideo.studio/assets/similarvid2input.webp">     
  </figure> 
  <figcaption>원본.</figcaption>  
</div>

<div style="text-align: center;">   
  <figure>     
    <img src="https://makeavideo.studio/assets/similarvid2c.webp">     
  </figure> 
  <figcaption>생성1.</figcaption>  
</div>



<div style="text-align: center;">   
  <figure>     
    <img src="https://makeavideo.studio/assets/i2vpair1.png">     
  </figure> 
  <figcaption>이미지 2개 interpolation.</figcaption>  
</div>

<div style="text-align: center;">   
  <figure>     
    <img src="https://makeavideo.studio/assets/i2vpair1.webp">     
  </figure> 
  <figcaption>이미지 2개 interpolation.</figcaption>  
</div>

# 4. Discussion

1. **인간 지능의 모방**:
   - 인간 지능의 가장 큰 장점 중 하나는 주변 세계로부터 배우는 능력이다.
   - 사람, 장소, 물건, 행동을 관찰을 통해 빠르게 인식하듯, 생성 시스템도 인간의 학습 방식을 모방하면 더 창의적이고 유용해질 수 있다.
2. **비지도 학습과 동영상 데이터의 활용**:
   - 대규모 비디오 데이터를 사용한 비지도 학습은 연구자들이 라벨링된 데이터에 대한 의존성에서 벗어나는 데 도움이 된다.
   - 라벨링된 이미지와 라벨 없는 비디오를 효과적으로 결합함으로써 이를 달성할 수 있음을 보여줌.
3. **기술적 한계 및 향후 계획**:
   - 현재 접근 방식은 텍스트와 비디오에서만 추론될 수 있는 현상 사이의 연관성을 학습하지 못한다.
   - 향후 작업으로는 왼쪽에서 오른쪽으로 손을 흔드는 사람과 같은 비디오를 생성하는 방법, 더 긴 비디오 생성, 여러 장면과 이벤트를 포함한 더 자세한 스토리를 묘사하는 방법 등을 고려 중.
4. **사회적 편향 및 윤리적 고려 사항**:
   - 웹 데이터로 훈련된 대규모 모델들은 사회적 편견을 학습하고 과장할 수 있으며, 해로운 것을 포함할 수 있다.



# 5. 개인 리뷰

논문 비약과 생략이 너무 많아서 간단하지만 생각보다 읽기는 어려웠음.

학습 할 때 비디오의 첫 프레임에 대한 것을 기반으로 text-prompt 및 image embedding을 진행하기 때문에 장면이 확 바뀌거나 영상 내부 복잡한  semantic이 존재하는 주제에 대해서는 비디오 생성을 하기 힘들 듯하다.

* 다 좋았지만 이 부분이 제일 아쉬웠던 부분.

[ControlNet](https://github.com/lllyasviel/ControlNet) 과 결합한다면 훨씬 더 복잡하고, 다양성 있는 비디오 생성이 가능할 것 같은 생각이 들었음.

 































