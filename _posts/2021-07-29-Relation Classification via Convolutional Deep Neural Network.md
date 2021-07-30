---
layout: post
title:  "[논문읽기]Relation Classification via Convolutional Deep Neural Network"
categories: ML AI Paper_Review
date:   2021-07-29 00:10:18 +0900
tags: Paper ConvNet Relation_Classification NLP
mathjax: True
author: Haribo
---
* content
{:toc}
> 여러가지 NLP process 중 하나이면서 정말 중요한 Relation Classification(Relation Extraction)을 1D Convoluton filter를 이용하여 해결하는 방법을 제시했던 논문이다.









# Reference

* [Relation Classification via Convolutional Deep Neural Network](https://aclanthology.org/C14-1220.pdf)

# Relation Classification

Relation Classification 또는 Relation Extraction*(이하 RE, 사실 두개가 정확히 같은것인지는 모르겠음)*문제는 여러 NLP 문제(감성분석, S2S 등등) 중 하나이다. 처음보는 분야라 감이 안잡혀서 5~10분정도 분량의 간략한 블로그글이나 설명을 보고싶었는데 아쉽게도 그러한 포스트가 존재하지 않았다. 앞으로 이와 관련된 논문을 여러편 더 읽어야 RE에 대한 이해가 완벽해 지겠지만, 우선 이 논문을 읽은 이해를 바탕으로 간략하게 RE문제를 요약하고 본격적으로 논문 리뷰를 해보겠다. RE문제는 **한 문장에서(또는 문단)에서 두개(혹은 그 이상)의 명사들간의 관계를 맞추는 문제이다.**

> The [fire]$$_{e_1}$$ inside WTC was caused by exploding [fuel]$$_{e_2}$$

이러한 문장이 있을 때 $e_1$과 $e_2$의 관계는 인과관계이다. 이렇게 문장내에서 두개의 명사간의 관계(Cause-Effect, Component-Whole, Entity-Origin 등등) 를 맞추도록 학습시키는 것이 바로 RE문제이다. 학습 데이터도 직관적이지 않고 좀 복잡하게 생겨서(나도아직 실습 안해봄) 이해하기 어렵다. 이 논문에서는 이러한 RE문제를 

> given a sentence $S$ with the annotated paris of nomials $e_1$ and $e_2$, we aim to identify the relations between $e_1$ and $e_2$

라고 정의하고있다. Vision 분야는 문제해결에 대부분 CNN을 기반으로 접근하는 반면에 이 문제해결에는 2가지 접근방법이 있다고한다.

> * Feature-Base
> * Kernel-Base

이 논문은 Feature-Base 방법으로 RE문제를 해결했다. Feature-Base 방법은 우리가 흔히아는 Feature(단어의 밀집벡터, input 이미지 등등)를 input으로 모델을 학습시켜 문제해결을 하는것을 의미한다. 그러나 Kernel-Base는 사실 모르겠다. 예전 SVM공부할 때, *...decision boundary를 좀 유연하게 긋기 위해서 input의 차원을 더 높은 차원에 projection시켜서 고차원에서 boundary 긋고 다시 차원을 원상복구하는...*, 이러한 차원을 갖고놀게 해주는 함수가 kernel함수 인것은 아는데 이게 어떻게 쓰이는지 이걸로 어떻게 모델을 구성하는지는 모르겠다. Kernel-Base RE문제 해결 논문을 보고 싶으면 [여기](https://www.jmlr.org/papers/volume3/zelenko03a/zelenko03a.pdf)를 보길 바란다. 진짜 어렵다.(kernel method에 대한 간략한 [설명](https://process-mining.tistory.com/95)). 

# Introduction

위에서 말했듯이 이 논문은 RE문제를 Feature-Base로 문제를 해결했다. 늘 그렇듯이 RE에 대한 기존의 Feature-Base 방법들을 까면서 시작하는데 한마디로 요약하자면 **"Raw한 문장(Sequence)에서 학습을 위해 Feature뽑아야하는데 잘 뽑으면 좋은 성능이 나오지만 잘못뽑으면 성능이 매우 안좋아진다."** 라고 말하고있다. 나도 RE 논문이 이게 처음이라 이 논문이전의 RE 트렌드에대해 모르지만 RE를 위해 Feature를 뽑는 전처리가 매우 어렵고 힘들다라는것을 유추해볼 수 있다. 이 논문이 주장하는 바는 "우리는 모델 외적으로 Feature를 뽑기위한 전처리 필요없다. 모델내부에서 해결한다. 그리고 성능도 더 좋다." 이렇게 요약할 수 있다.

# Architecture

![Architecture](/images/RE_Conv/framework.png)

이 모델을 간략하게 설명하자면

> 1. Raw한 input 문장이 들어온다.
> 2. input의 각 string을 밀집벡터(임베딩벡터)로 mapping 시켜준다.
> 3. 밀집벡터에서 2개의 정보를 뽑는다.
>    1. Lexical level features
>    2. Sentence level features
> 4. 두개의 Feature들을 concat해준다.
> 5. Softmax를 이용해 출력층을 Relation 갯수만큼의 확률분포로 만들어준다.

3단계가 액기스이고 그게 전부이기 때문에 이것만 이해하면 된다.

cont...