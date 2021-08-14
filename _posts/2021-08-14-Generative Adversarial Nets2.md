---
layout: post
title:  "[논문리뷰]Generative Adversarial Nets part2"
categories: ML AI Paper_Review
date:   2021-08-14 00:15:18 +0900
tags: Paper GAN
mathjax: True
author: Haribo
---
* content
{:toc}
> 지난 포스트에서 GAN의 아키텍처부분을 살펴보았고 이번에는 수학적으로 왜 GAN이 잘 동작하는지에대해 증명하는 부분과 실험결과 및 결론에대해서 마무리해보겠다.







# Reference

* [GAN 논문](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
* [GAN tutorial](https://ws-choi.github.io/blog-kor/seminar/tutorial/mnist/pytorch/gan/GAN-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC/)
* [GAN 리뷰 포스트 1](https://tobigs.gitbook.io/tobigs/deep-learning/computer-vision/gan-generative-adversarial-network)
* [GAN 리뷰 포스트 2](https://velog.io/@changdaeoh/Generative-Adversarial-Nets-GAN)
* [십분딥러닝-GAN](https://www.youtube.com/watch?v=0MxvAh_HMdY)
* [배경지식(커널 밀도추정)](https://jayhey.github.io/novelty%20detection/2017/11/08/Novelty_detection_Kernel/)
* [배경지식(볼츠만 머신)](https://horizon.kias.re.kr/18001/)
* [GAN 의 한계](http://dl-ai.blogspot.com/2017/08/gan-problems.html)





# Background

[지난 포스트](https://gkalstn000.github.io/2021/08/13/Generative-Adversarial-Nets1/) 초반부에 잠깐 언급했었던 두 확률분포가 얼마다 다른지를 알 수 있는 측도함수 중 하나인 [JSD](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)(Jensen–Shannon divergence)에 대해서 알아보고 넘어가겠다. JSD를 보기 이전에 정보량과 엔트로피에대해 알고 있어야하는데 모르면 [여기](https://ratsgo.github.io/statistics/2017/09/22/information/)를 참고하길 바란다. 

> $D_{JS}(P \parallel Q) = \frac{1}{2}D_{KL}(P \parallel M) + \frac{1}{2}D_{KL}(Q \parallel M),\ where\ M = \frac{P+Q}{2} $
>
> $D_{KL}(P \parallel Q) = \sum_{i}^{n}P(i)\log\frac{P(i)}{Q(i)} = \mathbb{E}_{x\sim p}\left [ \log\frac{P(x)}{Q(x)} \right ]$

JSD는 두 Kullback–Leibler divergence(이하 [KLD](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)) 의 평균이다. 엔트로피는 정규분포의 평균, 카이제곱 분포의 자유도 처럼 어떤 임의의 확률분포의 모수(parameter)의 역할을 할 수 있는데 KLD는 비교하고자 하는 확률분포($Q$)가 비교대상인 확률분포($P$)의 엔트로피를 얼마나 잘 보존하고 있는지의 척도이다. KLD 는 $[0, \infty ]$ 범위를 가지는데 두 확률분포 $P$와 $Q$가 완전히 같으면 0, 완전히 다르다면 $\infty$ 가 된다. 하지만 식에서 볼 수 있드시 KLD는 대칭이 성립하지 않는다

> $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$

그래서 KLD에 대칭성을 주기위한 거리측도함수가 바로 JSD이다. JSD는 $[0, 1 ]$ 범위를 가지는데 두 확률분포가 완전히 같으면 0, 완전히 다르면 1이 된다. 이전 포스트에서 GAN은 JSD를 이용해 두 확률분포의 차이를 알아낸다고 했는데 이번 포스트에서 아래의 GAN Loss 식이 사실 JSD를 기반으로한 함수 였다는 것을 증명할 것이다.

> $$
> \underset{G}{min}\ \underset{D}{max}V(D, G) = \mathbb{E}_{x\sim p_{data}(x)}[logD(x)]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z)))]
> $$



# 4. Theoretical Results

이론적 증명을 하기전에 논문에서 친절하게 GAN 학습과정을 직관적으로 보여주기위해 만든 figure가 있는데 그것부터 먼저 보고 넘어가겠다.

## figure 1

> 검은 점선 : $p_{data}$(원본 데이터의 확률분포)
>
> 초록선 : $p_{g}$(가짜 데이터의 확률분포)
>
> 파란점선 : $D$의 확률 분포
>
> $Z$ : latent space

![image-20210814161702023](/images/GAN/figure1.png)

![](/images/GAN/mapping.png)

중간에 이 그림이 있었으면 더 이해하기 좋았을텐데, 각 그림이 나타내는 것은 `a -> b -> c -> d`의 시간 순으로 latent space(noise)의 변수 $Z$가 $X$ 공간으로 mapping 시켜주는 확률변수 $G$에 의해서 변환된 데이터(가짜)들의 분포 $p_{g}$(초록선)가 갈수록 원본 데이터의 분포 $p_{data}$에 근사하게 되고 마지막엔 $p_{data} = p_{g}$가 되어서 $D$는 어떤 데이터가 들어오든 $\frac{1}{2}$ 확률로 진짜와 가짜를 구별하게 된다라는 스토리를 가진 figure다.

> (a) : 초반학습 단계가 지난 뒤, $D$는 완벽하진 않지만 어느정도 제 역할을 해내는 모습이고, $p_{g}$는 어느정도 $p_{data}$에 근사된 모습이다.
>
> (b) : 학습을 통해 $D \rightarrow  D^{\*}$로 수렴하게됨. $D^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)}$
>
> (c) : $D$의 optimal인 $D^{\*}$로부터 gradient 값을 받으며 $p_{g}$는 점차 $p_{data}$에 근사하게됨
>
> (d) : $p_{data} = p_{g}$ 가 되어서 $D$가 everywhere에서 $D(x) = \frac{1}{2}$ 이 된 모습

## Algorithm 1

![image-20210814181202898](/images/GAN/algorithm.png)

본격적으로 증명에 들어간다. Section 4는 Algorithm 1 이 이론적으로 $p_{g}$를 $p_{data}$ 에 수렴(근사)시킬 수 있다는 것을 증명하고자 한다.

* Section 4.1 : minmax game을 기반한 학습이 $p_{g} = p_{data}$로 수렴하는 global optimum 이 존재함을 증명

* Section 4.2 : Algorithm 1이 $$\underset{G}{min}\ \underset{D}{max}V(D, G) = \mathbb{E}_{x\sim p_{data}(x)}[logD(x)]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z)))]$$을 최적화 함을 증명

GAN의 Loss 함수를 보면 $D$가 $G$에게 어떤식으로 학습을 해야하는지 gradient를 준다. 즉 $D$ 는 $G$를 지도하는 선생님 역할을 하는 것이다. 이론적으로 완벽한 선생님에게 지도받는 학생은 선생님 만큼 완벽질 수 있다. GAN 논문에서 minmax game이란, 최악의 상황에서 손실을 최소화시키는, 즉 $p_{data}$의 분포를 완벽하게 학습한 $D$가 주는 피드백을 받고 $G$가 학습되는 상황을 말한다. 하지만 이것은 이론적으로나 가능한 일이지 $G$가 학습되기 이전에 $D$는 완벽하게 학습될 수 없다. $D$가 $G$에게 피드백을 주는만큼, $G$도 $D$에게 $p_{g}$에대한 정보를 제공하기 때문이다. 하지만 이 증명은 **이론적으로 완벽한 $D$, 즉 $D^{\*}$ 가 있다는 가정하에 $p_{g}$는 $p_{data}$에 수렴할 수 있다** 라는 것을 증명한다.



## 4.1 Global Optimality of $p_{g} = p_{data}$

### Proposition 1.

Optimal discriminator $D$. 말 그대로 최적의 discriminator를 의미한다. 어떠한 $G$가 오던간에, $G$ 상태가 어떻건 간에 $p_{data}$를 완벽히 학습한 $D$가 있다고 가정한다면 우리는 그 $D$를 $D_{G}^{\*}$라고 표기하고 그 값은 $\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$ 가 된다.

> $$D_{G}^{*}(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$$

### proof

어떠한 $G$가 오던간에 최적의 $D$를 만드든 일은 $V(G, D)$를 최대화 하는 것이다.

![image-20210814231835322](/images/GAN/proof1.png)

$V(D, G)$ 가 최대가 되기 위해서는 아래의 식이 최대가 되어야하는데 이는 미분을 통해 $V(D, G)$ 가 최대가되는 $D$를 구할 수 있다.

> $$p_{data}(x)\log(D(x)) + p_{g}(x)\log(1-D(x))$$

$a, b \geq 0,\ (a, b)\in \mathbb{R}^{2}$ 일 때 $a\log(y)+b\log(1-y)$ 꼴의 형태의 그래프는 아래의 그림과 같은 형태로서 최대값이 존재한다. 따라서 미분을 통해 $V(D, G)$가 최대값을 갖게하는 $D$는 $$D_{G}^{*}(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$$가 됨을 알 수 있다. 그런데 논문에 등장하는 $Supp$ 는 위상수학용어인 지지집합이라고 하는데 나도 위상수학을 몰라서 자세한건 모르겠다. 알고싶다면 [여기](https://ko.wikipedia.org/wiki/%EC%A7%80%EC%A7%80%EC%A7%91%ED%95%A9)를 들어가보길 바란다.

![image-20210814233239744](/images/GAN/proof2.png)

그런데 잘 보면 $D$의 trainning의 Loss함수는 MLE로 해석할 수 있다. MLE란 주어진 데이터를 바탕으로 그 데이터를 가장 잘 표현하는 확률분포를 찾는 것인데

> $$
> P(Y = y \mid x)
> \left\{\begin{matrix}
> y = 1\ if\ x\in p_{data} \\
> y = 0\ if\ x \in  p_{g}
> \end{matrix}\right.
> $$

즉 MLE를 통해 어떤 $x$가 input으로 왔을 때 $p_{data}$ 나 $p_{g}$ 에 따라 다른 확률값을 뱉는 확률분포를 추정하는 것으로 볼 수 있다. 그래서 다시 $V(D, G)$를 재구성 해보자.

![image-20210815003454276](/images/GAN/proof3.png)

위 식은 임의의 $G$를 고정시켜놓고 최적의 $D$를 대입해서 식을 고친 것이다.

### Theorem 1. 

>  $p_{g} = p_{data}$ 일 때 $C(G)$가 최소가 되고 그 때 $C(G)$는 $-\log4$의 값을 가진다.

### proof.

논문에서는 주저리주저리 써놨는데 식 유도한번으로 모든게 증명된다.

![image-20210814233239744](/images/GAN/proof4.png)

즉 GAN의 Loss 함수는 $p_{daga}$와 $p_{g}$ 사이의 거리를 줄여나가는 방식으로 학습이 되도록 설계 되었고, 그 거리를 측정하는 방식은 바로 JSD 였던 것이다. $p_{data} = p_{g}$ 인 경우 JSD는 0이 되므로 $C(G)$의 최소값은 $-\log4$가 된다.

## 4.2 Convergence of Algorithm 1

이 증명은 Algorithm 1이 진짜로 잘 동작하는지 증명하는 내용이다. 

### Proposition 2.

> $D$와 $G$의 성능이 충분하다면 Algorithm 1은 $G$가 주어졌을 때 $D \rightarrow D^{\*}_{G}$ 로 가고 $p_{g}$가 업데이트되면서 $p_{data}$로 수렴하게 된다.

### proof



# 5. Experiments

# 6. Advantages and disadvantages

# 7. Conclusions and future work

## Acknowledgments

