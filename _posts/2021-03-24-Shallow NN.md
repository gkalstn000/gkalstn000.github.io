---
layout: post
title:  "Shallow NN"
categories: ML AI
date:   2021-03-24 15:10:18 +0900
tags: Math
mathjax: True
author: Haribo
---
* content
{:toc}
## Shallow NN with Logistic Regression

`Neural Network`의 작동 원리를 알아보기위해 `Logistic Regression`에 **hidden layer**를 1층 추가해본다. 이 hidden layer를 추가함으로서 기존의 `Logistic Regression`으로 판별하지 못했던 `XOR` problem도 해결 할 수 있게된다. 그 이유는 hidden layer가 추가하게되면  **구별의 기준이 되는 hyper plane을 여러개를 사용하게 되서  이분법적인 예측을 하지않고 더욱 유연하게 예측을 할 수 있게 된다.**

![](/images/SNN/1.png)









$h$개의 노드로 이루어진 hidden layer를 추가 함으로서 조금 복잡해 졌지만 Logistic Regression을 여러개붙인것과 같다. 아래 그림의 빨간 동그라미 부분이 Logistic Regression 모델하나이다.

![](/images/SNN/2.png)

---

각 머신의 내부를 보면 $input$ 과 $parameter$의 내적을 해준다음 $activate$ 함수의 출력 결과물을 내어준다.

![](/images/SNN/3.png)

> $$
> z = 
> \begin{vmatrix}
> w_1 & w_2 & w_3 & \cdots w_m
> \end{vmatrix}
> \cdot
> \begin{vmatrix}
> x_1\\ 
> x_2\\ 
> x_3\\ 
> \vdots\\
> x_m
> \end{vmatrix} 
> +b
> $$

0층($input layer$)에서 1층($hidden layer$)으로의 수식은

![](/images/SNN/4.png)

>$$
>Z^{1} = 
>\begin{vmatrix}
>w^{1}_{11} & w^{1}_{12} & w^{1}_{13} & \cdots w^{1}_{1m}\\
>w^{1}_{21} & w^{1}_{22} & w^{1}_{23} & \cdots w^{1}_{2m}\\
>w^{1}_{31} & w^{1}_{32} & w^{1}_{33} & \cdots w^{1}_{3m}\\
>\vdots\\
>w^{1}_{h1} & w^{1}_{h2} & w^{1}_{h3} & \cdots w^{1}_{hm}\\
>\end{vmatrix}
>\cdot 
>\begin{vmatrix}
>x_1\\ 
>x_2\\ 
>x_3\\ 
>\vdots\\
>x_m
>\end{vmatrix} 
>+
>\begin{vmatrix}
>b^{1}_1\\ 
>b^{1}_2\\ 
>b^{1}_3\\ 
>\vdots\\
>b^{1}_h
>\end{vmatrix}
>= W^{1}X + b^{1}\\
>$$
>
>$$
>A^{1} = activate(Z^{1})
>$$

1층($hidden layer$)에서 2층($output layer$)으로의 수식은

![](/images/SNN/5.png)

>$$
>Z^{2} = 
>\begin{vmatrix}
>w^{2}_1 & w^{2}_2 & w^{2}_3 & \cdots w^{2}_h
>\end{vmatrix}
>\cdot
>\begin{vmatrix}
>a^{1}_1\\ 
>a^{1}_2\\ 
>a^{1}_3\\ 
>\vdots\\
>a^{1}_h
>\end{vmatrix} 
>+b^{2}
>= 
>W^{2}A^{1}+b^{2}\\
>$$
>
>$$
>A^{2} = activate(Z^{2})
>$$

## forward propagation

![](/images/SNN/6.png)

$activate\,function$과 $loss\,function$은 각각 종류가 여러개기 때문에 각 함수에 맞는 계산을 해주면 된다. 이번에 쓸 함수들은

> ### Activate function
>
> * $\varphi (Z^{1}) = tanh(Z^{1})$
> * $\varphi (Z^{2}) = \frac{1}{1+e^{-Z^{2}}}$
>
> ### Loss function
>
> * $L(y, A^{2}) = -\sum_{i=1}^{h}y_{i}lnA^{2}+(1-y_{i})ln(1-A^{2})$
>   * 손실함수는 최소를 만드는 것이 목적이기 때문에 MLE에 음수를 붙여 최소를 구하는 함수로 만들어 준다.

![](/images/SNN/7.png)

## backward propagation

parameter($W, b$)값을 변화시키면 마치 도미노 처럼 $Z^1, A^1, Z^2, A^2$ 그리고 최종적으로 $L(y, A^2)$ 값이 변한다. 그렇다면 parameter값들을 어떻게 변화시켜야 $L(y, A^2)$값이 작게 바뀔까? 바로 미분을 이용한다.

> $y\,=\,3x$ 인 방정식이 있을 때 $\frac{dy}{dx} = 3$이다. 이것은 $x$의 변화량에 따른 $y$ 변화량 수치를 나타내는 것인데, $x$가 1이 커지면 $y$는 3만큼 커지게되고, $x$가 1.5만큼 커지면 $y$는 4.5만큼 커진다.

손실함수의 값이 작아지는 방향으로 파라미터값을 변화시켜나가야 하기 때문에 파라미터($W, b$) 값들에 대한 $L(y, A^2)$의 **음의 그래디언트**방향으로 파라미터 값을 갱신해나가야 한다.

![](/images/SNN/8.png)

> $\frac{L(y, A^2)}{\partial W^2} = \frac{L(y, A^2)}{\partial A^2}\cdot \frac{A^2}{\partial Z^2}\cdot \frac{Z^2}{\partial W^2}$
>
> $\frac{L(y, A^2)}{\partial b^2} = \frac{L(y, A^2)}{\partial A^2}\cdot \frac{A^2}{\partial Z^2}\cdot \frac{Z^2}{\partial b^2}$
>
> $\frac{L(y, A^2)}{\partial W^1} = \frac{L(y, A^2)}{\partial A^2}\cdot \frac{A^2}{\partial Z^2}\cdot \frac{Z^2}{\partial A^1}\cdot \frac{A^1}{\partial Z^1}\cdot \frac{Z^1}{\partial W^1}$
>
> $\frac{L(y, A^2)}{\partial b^1} = \frac{L(y, A^2)}{\partial A^2}\cdot \frac{A^2}{\partial Z^2}\cdot \frac{Z^2}{\partial A^1}\cdot \frac{A^1}{\partial Z^1}\cdot \frac{Z^1}{\partial b^1}$

$\frac{\partial Z^1}{\partial W^1}$의  결과가 $h$개의 $X$인 이유는 

![](/images/SNN/2.png)

![](/images/SNN/9.png)

$hidden\,layer$의 노드들이 각 예측 모델이기 때문에 $h$개의 각 $\frac{\partial L(y,A^2)}{\partial Z^1}$의 원소하나를  $W$의 한 행에 전달해야하기 때문이다.

>  $\frac{\partial L(y,A^2)}{\partial W^2} = A^1(A^2-y)$
>
>  $\frac{\partial L(y,A^2)}{\partial b^2} = (A^2-y)$
>
> $\frac{\partial L(y,A^2)}{\partial W^1} = X(1-(A^1_{j})^2) W^1_j (A^2-y)$
>
> $\frac{\partial L(y,A^2)}{\partial b^1} = (1-(A^1_{j})^2) W^1_j (A^2-y)$
>
> `j : 1, 2, ..., h`

## 구현

```python
import matplotlib.pyplot as plt
import numpy as np
class LR :
    def __init__(self) :
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss = []
    def forward1(self, X):
        return np.dot(X, self.W1) + self.b1
    def forward2(self, X) :
        return np.dot(X, self.W2) + self.b2
    
    def back2(self, x, err):
        w_grad = err * x
        b_grad = err 
        return w_grad, b_grad
    def back1(self, x, err):
        w_grad = np.array([x*err_ for err_ in err])
        b_grad = err 
        return w_grad.T, b_grad
    def sigmoid(self, z) :
        return 1/(1 + np.exp(-z))
    def tanh(self, z) :
        return np.tanh(z)
    
    def fit(self, X, Y, hidden = 4, epochs=10000, lr = 0.005):
        self.W1 = np.random.normal(size = (X.shape[1], hidden))
        self.b1 = np.random.normal(size = hidden)
        self.W2 = np.random.normal(size = hidden)
        self.b2 = np.random.normal()
        
        for _ in range(epochs):
            for x, y in zip(X, Y):
                z1 = self.forward1(x) 
                a1 = self.tanh(z1)
                z2 = self.forward2(a1)
                a2 = self.sigmoid(z2)
                err2 = -(y - a2)
                err1 = (1-a1**2)*self.W2 * err2
                w2_grad, b2_grad = self.back2(a1, err2)
                w1_grad, b1_grad = self.back1(x, err1)
                
                self.W2 -= lr*w2_grad/(m:=len(X))
                self.b2 -= lr*b1_grad/m
                self.W1 -= lr * w1_grad/m
                self.b1 -= lr * b1_grad/m
            self.loss.append(err1.mean())
    def predict(self, X):
        z1 = self.forward1(X)
        a1 = self.tanh(z1)
        z2 = np.dot(a1, self.W2.T)
        return [1 if self.sigmoid(z) >= 0.5 else 0 for z in z2]        
        
    
    def lossgraph(self) :
        x = range(len(self.loss))
        plt.plot(x, self.loss, 'ro')
```

## XOR test

```
epoch = 10000
lr = 0.005
```

로 돌렸을 때 가끔 로스값이 튀어서 예측을 못하는 경우도 있지만 그럭저럭 잘 예측한다.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
model = LR()
model.fit(X, y)
```

```python
model.predict(X)
```

```
[0, 1, 1, 0]
```

```python
#loss graph
model.lossgraph()
```

![](/images/SNN/loss1.png)

![](/images/SNN/loss2.png)

![](/images/SNN/loss3.png)

![](/images/SNN/loss4.png)

