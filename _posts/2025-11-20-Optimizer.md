---
layout: post
title:  "Optimizer 반파시키기"
categories: 실습
date: 2025-11-20 11:40:18 +0900
tags: AI Optimizer 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

아래의 그래프들은 LoRA 학습할 때 내가 겪은 다양한 [Prodigy Optimizer](https://arxiv.org/pdf/2306.06101)의 lr 스케줄링 변화 그래프다.

![LoRA alpha 차이](/images/ai_techs/lora_comp_1.png)
![LoRA rank 차이](/images/ai_techs/lora_comp_2.png)
![LoRA apply 차이](/images/ai_techs/lora_comp_3.png)
![LoRA init method 차이](/images/ai_techs/lora_comp_4.png)







[Omini-control](https://github.com/Yuanshi9815/OminiControl) 만지며 처음알게 된 optimizer로 LR 스케줄링을 지가 알아서 해준다 (삼성과 메타 직원이 저자임 ㄷㄷ;). 나의 스승 kohya는 Diffusion 학습할 때 `adafactor`, `adamw8bit` 을 애용하는듯 하지만, prodigy를 써본 후 lr을 수동으로 조절해줘야하는 저 optimizer들은 사용하기 겁난다.  
그러나 뭔가 저 lr 이 올라가는 기준을 모르니 왠지모르게 답답하고, 완벽하게 이용하지 못하고 있는듯하다. 그래서 ChatGPT한테 알려달라했었는데 optimizer 기본 개념이 너무 부족하다보니 하나도 이해를 못했다 (논문 수식도 상당히 빡새보여서 보려면 1달정도 잡아야함).  
따라서 학부 때 대충 파악하고 (사실 거의 모멘텀 적용 optimizer부터 이해못함) 넘어갔던 optimizer 개념을 좀 잡은 후 `prodigy`, `adamW`, `adafactor`, `adamW8bit` 등 Diffusion 학습에 자주 보이는 optimizer를 파보려고한다.

# Optimizer 와 Loss의 관계
![Loss 와 Optimizer](/images/ai_techs/loss.png)
우선 현재 내 머리속에 들어있는 Loss와 optimizer의 관계다. likelihood 기반 loss 기준으로 $\theta$를 어떻게 만져나가야 loss를 줄일 수 있을지, $\theta$의 변화 방향을 제시하는 세르파가 optimizer다. 
* 근데 좀 의문인점이 보통 이 $\theta$ surface를 convex로 잡고 가던데, $\theta$가 최적점에 가까워질 때 loss가 작아져야만 그게 성립할텐데 무조건 성립하는가? 
* $$d(\theta^*, \theta_1) < d(\theta^*, \theta_2)$$ 일 때 $L(y, f_{\theta_1}(x)) > L(y, f_{\theta_2}(x))$ 이럴 경우는 절대 없는거? 아닐꺼같은데...
* 뭔가 머리속에서 Loss function이 어떤 특징을 지니냐에 따라 convex가 성립하냐 안하냐 달릴꺼같은데, 이건 수학이 너무 빡쌜꺼같음.

SGD 까지는 직관적으로 이해가 되었지만, 그 이후 모멘텀이 등장한 이후로 학부 때 제대로 이해 못하고 그런갑다 하고 넘어갔었다.

## 1. Momentum (1980s–1990s 전통 최적화에서 도입)
찾아보니 굉장이 유서 깊은 방식이었다. 핵심은 residual 느낌으로 이전 스텝에서 가고있던 방향에 현재 스텝에 구한 방향을 한스푼 첨가하는 방식이었다. 

$$
v_{t+1} = \beta v_t + (1-\beta)(-\nabla L(w_t)) \\
w_{t+1} = w_t + \eta v_{t+1}
$$

근데 살짝 헷갈렸던게 Loss 구할 때 내부에서 여태까지 계산된 모든 history (computational graph)가 같이 저장되는데 $v$도 그 그래프 내역이 저장되어있는건지 의문이 들었다. 그런데 $v_t$는 과거 gradient들의 가중 평균 state로 상수 역할을 한다고 한다.  
보통 $\beta$는 0.9를 둔다. 즉, 이전방향 0.9배에다가 현재 SGD로 구한 방향 0.1을 더해서 방향이 일정하면 가속이 계속 붙어서 모멘텀이라는 이름이 붙여졌다.  

**일단 $v$ 가지고 있어야하니 VRAM 2배로 먹음**

## 2. Nesterov Accelerated Gradient (NAG, 2000s)

$$
g_t = \nabla L(w_t + \beta v_t) \\
v_{t+1} = \beta v_t - \eta g_t
$$

얘는 너 어차피 그쪽 방향으로 $\beta$배 만큼 갈꺼잖아. 걍 거기서 gradient 계산해라. 이런 느낌이다.  
Momentum 방식이 loss surface가 valley 많은 곳에서 모멘텀 때문에 최적점을 지나쳐 튕겨나가는 케이스가 많이 존재한다고 한다. 
* 먼저 쎄게 밀고 나중에 gradient 계산 -> 내릴곳 지남...

Nesterov는 밀기 전에, 미리 밀린 위치에서 gradient를 보고 나서 업데이트하자는 얘기다. 그러면 적어도 튕겨나가지는 않을 테니까.

## 3. AdaGrad (2011)

기본 Gradient Descent는 모든 파라미터에 같은 learning rate를 씀

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

문제
 1.	어떤 파라미터는 gradient가 자주 크게 나옴
 2.	어떤 파라미터는 gradient가 거의 0에 가까움

특히 **sparse 데이터(NLP, word embedding)** 에서 문제 심각
 * 자주 등장하는 feature → gradient 자주 큼
 * 드물게 등장하는 feature → gradient 거의 없음

이 둘에 같은 learning rate 쓰면 너무 비효율적임.  

여기서 핵심 아이디어는 각 파라미터가 과거에 얼마나 gradient를 많이 받아왔는지를 가지고 있는것.

$$
g_i = \nabla L(w_i)  \\
G_t = \sum_{i=1}^{t} g_i^2
$$

파라미터별로 따로 저장됨. 즉, weight 하나마다 자기만의 G가 있음.

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

그래서 파라미터 업데이터에 자주 활용된 놈일수록 패널티가 많이 들어가버린다. 

## 4. RMSProp (2012, Hinton) 힌튼햄 입갤ㄷㄷ;
  
AdaGrad의 핵심 아이디어는
“파라미터마다 과거 gradient 크기를 누적해서 learning rate를 조절하자”였지만,

치명적인 문제 하나가 있음.
$$
G_t = \sum_{i=1}^{t} g_i^2
$$

이 값이 계속 누적되기만 해서 끝없이 커짐.
그러면 learning rate가 다음처럼 점점 0에 가까워져 버린다 (너무 자주쓰인다고 아예 못쓰게함ㅋ)

$\frac{1}{\sqrt{G_t}} \to 0$

훈련 중반부터 파라미터가 거의 안 움직여버리는 현상이 발생하는데 이게 AdaGrad가 초반에는 빠른데 금방 멈추는 이유임.  
힌튼햄은 이거를 EMA로 해결해버림.
* $g^2$ 을 그냥 계속 더하지 말고, 오래된 것은 점점 잊고, 최근 것에 더 가중치를 주자.

$$
g_t = \nabla L(w_t)  \\
s_t = \rho\, s_{t-1} + (1 - \rho) g_t^2 \\
w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t
$$

* $	\rho:$ 보통 0.9

Momentum 처럼 자주 활용되는 파라미터놈을 EMA로 처리해준다.

## 5. Adam (2015)
## 6. AdamW (2017)

# 최신 기법 (adafactor, adamW8bit, prodigy, Lion 등)
