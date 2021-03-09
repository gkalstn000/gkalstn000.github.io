---
layout: post
title:  "functools"
categories: 노트
date:   2021-03-09 19:10:18 +0900
tags: python_module
author: Haribo
---
* content
{:toc}
`functools` 모듈안에 함수가 꽤많지만 내가 판단했을 때, 코딩테스트에 필요한 함수 몇가지들만 추려보았다. 코딩테스트에는 딱히 객체지향적인 프로그램이 필요하지 않기 때문에, `class`에 관련된 함수는 제외하였다.

> * `lru_cache`
> * `reduce`









## `lru_cache(maxsize=128, typed=False)`

[공식 문서](https://docs.python.org/ko/3/library/functools.html#functools.lru_cache)

> 가장 최근의 *maxsize* 호출까지 저장하는 기억하는(memoizing) 콜러블 함수를 감싸는 데코레이터. 비싸거나 I/O 병목 함수가 같은 인자로 주기적으로 호출될 때 시간을 절약할 수 있습니다.
>
> * `maxsize` : `maxsize` 만큼 호출된 반환값을 저장
> * `typed` : 반환타입을 구별하여 저장한다.
>   * `1.0` 과 `1` 을 다른 값으로 판단해 구별할 수 있다.

데코레이터 `@`를 사용해 함수의 반환값을 메모제이션할 수 있다. 예를들면

```python
@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

`fib(i)` 값을 계산할 때 `lru_cache` 에 저장된 값을 이용해 더 빠르게 계산할 수 있다.

 나는 [짝수행 세기](https://gkalstn000.github.io/2021/02/02/%EC%A7%9D%EC%88%98-%ED%96%89-%EC%84%B8%EA%B8%B0/) 조합 부분을 `@lru_cache`로 바꿨는데 기존 `dictionary` 를 이용한 캐시보다 **1.3**배정도 빨라졌다. 코드 또한 훨씬 간결해짐

```python
# 기존 캐시
def C(n,k):
    if (n,k) not in cache:
        cache[(n,k)]= comb(n,k)
    return cache[(n,k)]

#lru_cache
@lru_cache(maxsize=None)
def C(n,k): return comb(n,k)
```

## ` reduce(function, iterable, initializer = None)`

[공식문서](https://docs.python.org/ko/3/library/functools.html#functools.reduce)

> 두 인자의 *function*을 왼쪽에서 오른쪽으로 *iterable*의 항목에 누적적으로 적용해서, 이터러블을 단일 값으로 줄입니다. 예를 들어, `reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])`는 `((((1+2)+3)+4)+5)`를 계산합니다. 왼쪽 인자 *x*는 누적값이고 오른쪽 인자 *y*는 *iterable*에서 온 갱신 값입니다. 선택적 *initializer*가 있으면, 계산에서 이터러블의 항목 앞에 배치되고, 이터러블이 비어있을 때 기본값의 역할을 합니다. *initializer*가 제공되지 않고 *iterable*에 하나의 항목만 포함되면, 첫 번째 항목이 반환됩니다.

`iterable`한 객체의 누적계산을 구하는 부분에서 `itertools.accumulate` 와 비슷하지만 `reduce`는 누적계산 함수를 원하는대로 지정할 수 있다.  

실제 `reduce` 함수 코드

```python
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
```

---

`reduce`를 이용한 피보나치 함수 구현

```python
from functools import reduce
fib = lambda n : reduce(lambda x, n :[x[1], x[0] + x[1]], range(n), [0, 1])[-1]
```



