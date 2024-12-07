---
layout: post
title:  "StableDiffusion 실습"
categories: 실습
date: 2023-12-15 11:40:18 +0900
tags: Diffusion 생성모델
mathjax: True
author: Haribo
---
* content
{:toc}


[Stable Diffusion](https://github.com/Stability-AI/stablediffusion) 

[Stable Diffusion2](https://github.com/Stability-AI/stablediffusion) 



# 0. 포스팅 목적

업계표준 large model framework 스타일 파악

멀티모달 데이터셋 생김새 파악

경험

등등



처음 보는거

* lightning
* Trainer
* callback
* checkpoint
  * forward에 있는거

# 1. LAION Dataset download

[LAION 5B](https://laion.ai/blog/laion-5b/) 는 용량이 2TB 이기 때문에 [LAION 400M](https://laion.ai/blog/laion-400-open-dataset/) 사용





5B 다 쓰기에는 너무 클 것 같아서 [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) 만 사용하기로 결정.



## 1.1 Downloading the metadata

**메타데이터란** 데이터셋의 각 이미지와 텍스트에 대한 정보를 포함한 데이터. 여기에는 이미지의 URL, 해상도, 라이선스 정보, 안전성 태그 등이 들어있다.

``` bash
mkdir laion2B-en && cd laion2B-en
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$i-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet; done
cd ..
```



## 1.2 Filtering the metadata

- **목적**: 필요한 데이터만 선택하기 위해 메타데이터를 필터링. 예를 들어, 특정 해상도 이상의 이미지만 선택하거나, 안전한 이미지만 선택.
- **방법**: `pyspark`를 사용하여 필터링을 수행.

pyspark install

```bash
pip install pyspark 
```

Filtering

```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import rand

def main():
  spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-stats').getOrCreate()
  df = spark.read.parquet("laion2B-en/part-*.snappy.parquet")

  df = df.filter((df.WIDTH >= 1024) & (df.HEIGHT >= 1024))
  df = df.orderBy(rand()) # this line is important to have a shuffled dataset
  df.repartition(128).write.parquet("laion2B_big")

main()
```

