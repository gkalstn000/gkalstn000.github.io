---
layout: post
title:  "JadenCase 문자열 만들기"
categories: 프로그래머스
date:   2021-01-21 11:40:18 +0800
tags: Lv.2
author: Haribo
---

* content
{:toc}
[JadenCase 문자열 만들기](https://school.programmers.co.kr/learn/courses/30/lessons/12951)

# 코드

```python
def solution(s):
    return ' '.join([word.capitalize() for word in s.split(" ")])
```

`capitalize` : 단어의 첫문자를 대문자로 바꿔주는 함수