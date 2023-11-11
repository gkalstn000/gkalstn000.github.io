---
layout: page
title: About
permalink: /about/
icon: heart
type: page
---

* content
{:toc}
# 하민수 (Min Su Ha)

<div style="text-align: left; width: 25%;">   
  <figure>     
    <img src="/images/profile/KakaoTalk_Photo_2023-10-30-00-50-55.jpeg" >     
  </figure> 
</div>

March, 28, 1995

대한민국, 서울특별시

-- -

## Skill

* Python
* Pytorch
* Deep Learning
* Generative AI
  * GAN
  * Diffusion
  * VAE
* Pose-Guided Person Image Generation

## Education

**Hanyang University / M.S.**

* 2022.03 ~ 2024.02
* Artificial Intelligence
* Advisor: Woohwan Jung

**Hanyang University ERICA / B.S.**

* 2014.03 ~ 2021.08
* Computer Science & Engineering



## Research

**자세기반 이미지 생성을 위한 SAPDE 정규화 구조 활용**

**세부 분류 개체명 인식을 위한 액티브 러닝 쿼리전략 분석** 

* 한국정보과학회 학술발표논문집 2019(KSC 2022) [[DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224196)]

**약한 지도 학습 기반 태양광 발전시설 고장 탐지**

* 한국정보과학회 학술발표논문집 2019 (KSC 2021) [[DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11035770)]



## Software Registration

**액티브러닝기반 NER 데이터셋 레이블링 프로그램**

* 제 C-2022-054280 호



## Project

### **Pose Guided Person Image Generation**

<div style="text-align: center;">   
  <figure>     
    <img src="/images/profile/image-20231031004656979.png" >     
  </figure> 
</div>

Pose Guided Person Image Generation (PGPIG) task는 source image의 texture, identity 등 특징을 유지하며 임의의 자세에 맞는 이미지 생성하는 task.

다양한 `GAN`, `Diffusion` 등 다양한 SOTA 연구에 아이디어 접목.

* DPTN (CVPR 2022), NTED (CVPR 2022), PIDM (CVPR 2023)

#### Technical Stack

* `Python`, `Pytorch`, `GAN`, `PGPIG`
* 프로젝트 주소
  * DPTN with SPADE [Github](https://github.com/gkalstn000/2-stage-PGPIG)
  * Step DPTN
    * [DPTN](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network) (CVPR 2022)
  * Step NTED [Github](https://github.com/gkalstn000/NTED_step)
    * [NTED](https://github.com/RenYurui/Neural-Texture-Extraction-Distribution) (CVPR 2022)

#### Major Project Highlights

* 기존 PGPIG model (DPTN)에 SPADE norm 활용을 통해 source image 특징을 intermediate feature map에 효과적으로 반영

* Coarse-texture 이미지에서 Fine-texture 이미지로 step-by-step 방식으로 이미지 생성

  * 기존 GAN 방식 PGPIG 모델이 하지 못하는 서로다른 두 이미지의 Style을 Mixing이 가능

  * <div style="text-align: center;">   
      <figure>     
        <img src="/images/profile/image-20231031010326447.png" >     
      </figure> 
    </div>

---

### **BirdsEye Billiards Converter**

<div class="responsive-video-container" style="text-align:center;">
  <iframe src="https://www.youtube.com/embed/Sc0SwqfaRpM" title="example" frameborder="0" style="margin: 0 auto; display: block;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
#### Technical Stack

* `Python`, `Tensorflow`, `OpenCV`, `Object Detection`, `Linear Transformation`
* 프로젝트 주소 [Github](https://github.com/gkalstn000/capstone)

#### Major Project Highlights

* Objection Detection을 활용한 당구대, 당구공 인식.

* 자체 선형 변환 알고리즘 개발

  * openCV [warpAffine](https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html) 변환 라이브러리보다 더 높은 정확도

  * 

  * <div style="text-align: center;">   
      <figure>     
        <img src="/images/profile/image-20231031003105820.png" >     
      </figure> 
    </div>

    <div class="responsive-video-container" style="text-align:center;">
    	<iframe src="https://www.youtube.com/embed/qnMxNfZViCY" title="" frameborder="0" style="margin: 0 auto; display: block;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
          <p style="text-align:center; font-size: 14px; margin-top: 10px;">좌: openCV warpAffine() 라이브러리, 우: 자체 개발 알고리즘</p>
    </div>

  * **좌**: openCV `warpAffine()` 라이브러리,     **우**: 자체 개발 선형 변환 알고리즘

---

### **Active Learning을 활용해 효율적인 NER 학습**

#### Technical Stack

* `Python`, `Pytorch`, `Active Learning`, `BALD`,   `NLP`, `Named Entity Recognition`

#### Major Project Highlights

* Active Learning은 모델이 라벨링이 없는(unlabeled) 데이터 중 가장 정보량이 많은 데이터를 선택해 사람에게 라벨링을 요청.
* 정보량이 높은 데이터일수록 모델은 해당 데이터의 정답(라벨)을 판단함에 있어 어려움이 있을 것.
* 모델이 분류한 데이터 중 score가 낮았던 데이터와 Active Learning에서 정보량이 높다고 판단했던 데이터가 서로 반비례 관계가 있을 것이라는 가설을 세웠고 실험을 통해 확인.
* ![image-20231031012952347](../images/profile/image-20231031012952347.png)
  * 막대 그래프는 active learning을 통해 얻은 불확실성 score, 파란 그래프는 entity 별 f1-score.
  * entity에 대한 불확실성과 f1-score는 서로 반비례 관계를가짐, active learning은 모델이 어려워하는 entity 위주로 데이터 선별 -> 효과적인 학습 가능.

---

### **Font-generation via Diffusion Model (진행 중...)**





## 연락 주소

* [github](https://github.com/gkalstn000){: target="_blank"}
* email：gkalstn000@gmail.com

## Comments

{% include comments.html %}
