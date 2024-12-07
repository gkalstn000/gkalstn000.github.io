---
layout: post
title:  "itertools"
categories: 노트
date:   2021-03-10 13:10:18 +0900
tags: python_module
mathjax: True
author: Haribo
---
* content
{:toc}
[itertools](https://docs.python.org/ko/3/library/itertools.html)는 버릴것 하나없는 아주 유용한 모듈이다.

> ### 무한 이터레이터
>
> * `count`
> * `cycle`
> * `repeat`
>
> ### 이터레이터
>
> * `accumulate`
> * `chain`
> * `chain.from_iterable`
> * `compress`
> * `dropwhile`
> * `filterfalse`
> * `groupby`
> * `islice`
> * `tee`
> * `zip_longest`
>
> ### 조합형
>
> * `product`
> * `permutations`
> * `combinations`
> * `combinations_with_replacement`









## 무한 이터레이터

무한 이터레이터기 때문에 탈출문이 필수다. 딱히 실전에 쓰일일은 없을듯

### **count**

`count(start=0, step=1)`

`range`와 비슷하지만 `step`에 소수점을 넣을 수 있음

```python
from itertools import count
for i in count(3, -0.2) :
    print(i)
    if i < 0 : break
```

---

### **cycle**

`cycle(iterable)`

`iterable`을 무한반복해준다.

```python
from itertools import cycle
for i in cycle('abcd') :
    print(i)
    if i == 'd' : break
```

---

### **repeat**

`repeat(object, times = None)`

`object`를 `times` 만큼 반복시킬 수 있음. `default`는 무한

```python
from itertools import repeat
list(map(pow, range(10), repeat(2)))
```

---

## 유한 이터레이터

가장 많이 쓰이고, 필요한 함수들

---

### **accumulate**

`accumulate(iterable, func=operator.add, *, initial=None)`

누적합 리스트를 구할 때 매우 유용하다

```python
from itertools import accumulate
import operator
data = [3, 4, 6, 2, 1, 9, 0, 7, 5, 8]
list(accumulate(data, operator.mul)) 
list(accumulate(data, operator.add)) 
```

---

### **chain.from_iterable**, **chain**

2차원배열을 1차월 배열로 변환시킬 때 매우 유용하다.

```python
from itertools import chain
a = [[1, 2, 3], [4, 5, 6]]
list(chain(*a))
list(chain.from_iterable(a))
```

둘다 결과가 같지만 굳이 고르자면 더 짧은 `chain` 쓰는게 덜 귀찮을듯

---

### **compress**

`compress(data, selectors)`

`selectors`로 원하는 값만 뽑을 수 있다. `flag` 기능

```python
from itertools import compress
''.join(compress('ABCDEF', [1,0,1,0,1,1]))
-->'ACEF'
```

---

### **dropwhile**, **takewhile**

`dropwhile(predicate, iterable)`

> `predicate`가 처음으로 거짓이 되는 순간 그 후의 요소들을 출력한다.

```python
from itertools import dropwhile
list(dropwhile(lambda x: x<5, [1,4,3,4,1, 1, 1, 1, 23, 4, 4]))
--> [23, 4, 4]
```

---

`takewhile(predicate, iterable)`

`dropwhile`과 반대.

```python
from itertools import dropwhile
list(dropwhile(lambda x: x<5, [1,4,3,4,1, 1, 1, 1, 23, 4, 4]))
--> [1,4,3,4,1, 1, 1, 1]
```

---

### **filterfalse**

`filterfalse(predicate, iterable)`

> `predicate`에서 거짓인 요소들만 출력해준다

```python
from itertools import filterfalse
filterfalse(lambda x: x%2, range(10)) 
--> 0 2 4 6 8
```

---

### **groupby**

`groupby(iterable, key = None)`

> `key` 값을 지정해주면 `key`값을 기준으로 요소들을 그룹핑해준다
>
> `key`값을 지정해주지 않으면 `iterable`의 현재요소가 `key`가 된다.

```python
# key 값 지정
from itertools import groupby
a_list = [("Animal", "cat"),  
          ("Animal", "dog"),  
          ("Bird", "peacock"),  
          ("Bird", "pigeon")] 
an_iterator = groupby(a_list, lambda x : x[0]) 
for key, group in an_iterator: 
    key_and_group = {key : list(group)} 
    print(key_and_group)
```

```
{'Animal': [('Animal', 'cat'), ('Animal', 'dog')]}
{'Bird': [('Bird', 'peacock'), ('Bird', 'pigeon')]}
```

---

```python
# key값 ㄴ지정
# 몇개의 덩어리가 있는지 알수있음
from itertools import groupby
a_list = 'aaaabbbccbbbaaaasss'
for key, group in groupby(a_list): 
    print(key, len(list(group)))     
```

```
a 4
b 3
c 2
b 3
a 4
s 3
```

---

### **islice**

`itertools.islice(iterable, stop)`

`itertools.islice(iterable, start, stop, step = None])`

인덱싱 함수. 굳이 쓸필요는 없을듯하다

```python
from itertools import islice
islice('ABCDEFG', 2) --> A B
islice('ABCDEFG', 2, 4) --> C D
islice('ABCDEFG', 2, None) --> C D E F G
islice('ABCDEFG', 0, None, 2) --> A C E G
```

---

### **tee**

`tee(iterable, n=2)`

> `iterable`의 복사본을 `n`개만큼 만든다

```python
import itertools   
iterator1, iterator2 = itertools.tee([1, 2, 3, 4, 5, 6, 7], 2) 
print(list(iterator1))  
print(list(iterator1))  
print(list(iterator1))  
print(list(iterator2))  
print(list(iterator2))  
```

```
[1, 2, 3, 4, 5, 6, 7]
[]
[]
[1, 2, 3, 4, 5, 6, 7]
```

복사본 2개가 각각 `iterator1`, `iterator2`에 하나씩 들어가서 한번씩 참조되면 사라짐. **원본이 먼저 참조되면 복사본은 전부 사라진다**

---

### **zip_longgest**

`zip_longest(*args, fillvalue=None)`

직접 보면 바로이해함

```python
from itertools import zip_longest
zip_longest('ABCD', 'xy', fillvalue='-') 
```

```
Ax By C- D-
```

갯수가 안맞으면 `fillvalue` 값으로 채움

## 조합형 이터레이터

코테에 아주 유용하고 안쓰면 병신인 함수들

### **product**

카디션곱 함수, 뿐만 아니라 `flag` 만들 수 있음

```python
product('ABCD', 'xy')
product(range(2), repeat=3)
```

```
Ax Ay Bx By Cx Cy Dx Dy
000 001 010 011 100 101 110 111
```

### **permutations**, **combinations**

순열과 조합

```python
from itertools import permutations, combinations
permutations('ABCD', 2)
combinations('ABCD', 2)
```

```
AB AC AD BA BC BD CA CB CD DA DB DC
AB AC AD BC BD CD
```

### **combinations_with_replacement**

중복조합 함수 $$_{n}\textrm{H}_{r}$$

```python
from itertools import combinations_with_replacement
list(combinations_with_replacement('ABC', 3))
```

```
[('A', 'A', 'A'),
 ('A', 'A', 'B'),
 ('A', 'A', 'C'),
 ('A', 'B', 'B'),
 ('A', 'B', 'C'),
 ('A', 'C', 'C'),
 ('B', 'B', 'B'),
 ('B', 'B', 'C'),
 ('B', 'C', 'C'),
 ('C', 'C', 'C')]
```

