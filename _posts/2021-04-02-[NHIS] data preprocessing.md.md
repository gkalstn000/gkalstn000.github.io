---
layout: post
title:  "[NHIS] data proprecessing"
categories: ML AI
date:   2021-04-02 15:10:18 +0900
tags: pre_process
mathjax: True
author: Haribo
---
* content
{:toc}
# feature decision

## Drop features

* `HCHK_YEAR` 
* `IDV_ID`
* `HCHK_OE_INSPEC_YN`
* `CRS_YN`
* `TTH_MSS_YN`
* `ODT_TRB_YN`
* `WSDM_DIS_YN`
* `TTR_YN`
* `DATA_STD__DT`

## Target

> 당뇨(**2**) : `126 <= BLDS`
>
> 전당뇨(**1**) : `100 <= BLDS < 126`
>
> 정상(**0**) : `BLDS < 100`

## Categorical features

* `SEX`
* `SIDO`
* `HEAR_LEFT`
* `HEAR_RIGHT`
* `SMK_STAT_TYPE_CD`
* `DRK_YN`

## Numercial features

* `AGE_GROUP`
* `HEIGHT`
* `WEIGHT`
* `BMI`
* `WAIST`
* `SIGHT_LEFT`
* `SIGHT_RIGHT`
* `OLIG_PROTE_CD`
* `BP_HIGH`
* `BP_LWST`
* `TOT_CHOLE`
* `TRIGLYCERIDE`
* `HDL_CHOLE`
* `LDL_CHOLE`
* `HMG`
* `CREATININE`
* `SGOT_AST`
* `SGPT_ALT`
* `GAMMA_GTP`

---


```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import scipy.stats as stats
from scipy.stats.mstats import winsorize
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# visualization
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# ignore warnings
import warnings
warnings.filterwarnings(action='ignore')


# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```


```python
data_2017 = pd.read_csv('./NHIS_2017_2018_100m/NHIS_OPEN_GJ_2017_100.csv')
data_2018 = pd.read_csv('./NHIS_2017_2018_100m/NHIS_OPEN_GJ_2018_100.csv')
df = pd.concat([data_2017,data_2018])
```

# Drop features

필요없는 `feature`들을 삭제하고 결측치 분포를 보니 `'TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE'`의 결측치가 가장 많고, 또한 결측치가 서로 몰려있는 것을 확인할 수 있다.


```python
drop_cols = ['HCHK_YEAR', 'IDV_ID', 'HCHK_OE_INSPEC_YN', 'CRS_YN', 'TTH_MSS_YN', 'ODT_TRB_YN', 'TTR_YN', 'WSDM_DIS_YN', 'DATA_STD__DT']
df.drop(drop_cols, axis = 1, inplace = True)
```


```python
# error시 : sudo conda install missingno
msno.matrix(df)
plt.show()
```


![png](/images/nhis_p/output_5_0.png)
    

# Target

당뇨병의 무서운점은 당뇨병 자체가 아니라 그로인해 발생하는 합병증 때문인다. 따라서 당뇨병은 조기진단이 매우 중요한 질병이다. 따라서 정상, 전당뇨, 당뇨 환자를 예측하는 Target을 설정한다.  
[전당뇨 정보](http://www.samsunghospital.com/home/healthInfo/content/contenView.do?CONT_SRC_ID=33901&CONT_SRC=HOMEPAGE&CONT_ID=6807&CONT_CLS_CD=001027)

>당뇨(**2**) : `126 <= BLDS`
>
>전당뇨(**1**) : `100 <= BLDS < 126`
>
>정상(**0**) : `BLDS < 100`


```python
sns.kdeplot(df.BLDS)
plt.title("BLDS distribution")
plt.show()
```


![png](/images/nhis_p/output_7_0.png)

```python
p = len(df[df['BLDS'] >= 126])
print('[공복혈당 126 이상]당뇨환자 수치 : {}%'.format(100 * p / len(df)))
p = len(df[(100 <= df['BLDS']) & (df['BLDS'] < 126)])
print('[공복혈당 100 이상 126 미만]전 당뇨환자 수치 : {}%'.format(100 * p / len(df)))
p = len(df[df['BLDS'] < 100])
print('[공복혈당 100 미만]정상인 수치 : {}%'.format(100 * p / len(df)))
```

    [공복혈당 126 이상]당뇨환자 수치 : 7.795%
    [공복혈당 100 이상 126 미만]전 당뇨환자 수치 : 30.21825%
    [공복혈당 100 미만]정상인 수치 : 61.6889%


```python
df.loc[ df['BLDS'] < 100, 'BLDS'] = 0
df.loc[(100 <= df['BLDS']) & (df['BLDS'] < 126), 'BLDS'] = 1
df.loc[126 <= df['BLDS'], 'BLDS'] = 2
df = df.dropna(subset = ['BLDS'], how = 'any', axis=0)
```


```python
msno.matrix(df)
plt.show()
print(df.isnull().sum())
```


![png](/images/nhis_p/output_11_0.png)
    


    SEX                      0
    AGE_GROUP                0
    SIDO                     0
    HEIGHT                   0
    WEIGHT                   0
    WAIST                  672
    SIGHT_LEFT             417
    SIGHT_RIGHT            434
    HEAR_LEFT              357
    HEAR_RIGHT             355
    BP_HIGH                 40
    BP_LWST                 40
    BLDS                     0
    TOT_CHOLE           661336
    TRIGLYCERIDE        661346
    HDL_CHOLE           661347
    LDL_CHOLE           671083
    HMG                     26
    OLIG_PROTE_CD         9274
    CREATININE               4
    SGOT_AST                 2
    SGPT_ALT                 3
    GAMMA_GTP                6
    SMK_STAT_TYPE_CD       377
    DRK_YN              350820
    dtype: int64


# Categorical features
범주형 데이터들은 모두 `one-hot-encoding` 해준다
* `SEX`
* `SIDO`
* `HEAR_LEFT`
* `HEAR_RIGHT`
* `SMK_STAT_TYPE_CD`
* `DRK_YN`

## SEX
성비 
* 남 : 여 = 1.4 : 1

남성이 여성보다 유병률이 높다  
* 남성 : 45%
* 여성 : 30%


```python
mapper = {1 : 'male', 2 : 'female'}
df['SEX'].replace(mapper, inplace=True)
df_ = df[["SEX", "BLDS"]]
df_['BLDS'] = np.where(df_['BLDS'] == 2, 1, df_['BLDS'])
df_.groupby(['SEX'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```

<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEX</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.443937</td>
    </tr>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.309574</td>
    </tr>
  </tbody>
</table>

</div>




```python
df['SEX'] = df['SEX'].astype('category')
pd.get_dummies(df['SEX'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1994043 rows × 1 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

## SIDO
식습관 때문인지 지역별로 당뇨 유병률에 차이가 있음을 알 수 있다.  
그런데 매뉴얼에 나와있지않은 도시코드(50)가 하나있는데 아마 외국인이 아닌가 싶다


```python
mapper = {11 : 'SEOUL',
          26 : 'BUSAN',
          27 : 'DAGU',
          28 : 'INCHEON',
          29 : 'KWANGJU',
          30 : 'DAJEON',
          31 : 'ULSAN',
          36 : 'SEJONG',
          41 : 'GYEONGGI',
          42 : 'GANGWON',
          43 : 'CB',
          44 : 'CN',
          45 : 'JB',
          46 : 'JN',
          47 : 'GB',
          48 : 'GN',
          49 : 'JEJU',
          50 : 'FOREIGN'}
df['SIDO'].replace(mapper, inplace=True)
df[["SIDO", "BLDS"]].groupby(['SIDO'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SIDO</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>JN</td>
      <td>0.567383</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KWANGJU</td>
      <td>0.503593</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GANGWON</td>
      <td>0.491400</td>
    </tr>
    <tr>
      <th>0</th>
      <td>BUSAN</td>
      <td>0.486423</td>
    </tr>
    <tr>
      <th>11</th>
      <td>JB</td>
      <td>0.476529</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CN</td>
      <td>0.476058</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GN</td>
      <td>0.466652</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GB</td>
      <td>0.464439</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FOREIGN</td>
      <td>0.460401</td>
    </tr>
    <tr>
      <th>10</th>
      <td>INCHEON</td>
      <td>0.460256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DAJEON</td>
      <td>0.453052</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CB</td>
      <td>0.451514</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GYEONGGI</td>
      <td>0.449345</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ULSAN</td>
      <td>0.435314</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SEOUL</td>
      <td>0.432545</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DAGU</td>
      <td>0.426531</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SEJONG</td>
      <td>0.412095</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['SIDO'] = df['SIDO'].astype('category')
pd.get_dummies(df['SIDO'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CB</th>
      <th>CN</th>
      <th>DAGU</th>
      <th>DAJEON</th>
      <th>FOREIGN</th>
      <th>GANGWON</th>
      <th>GB</th>
      <th>GN</th>
      <th>GYEONGGI</th>
      <th>INCHEON</th>
      <th>JB</th>
      <th>JN</th>
      <th>KWANGJU</th>
      <th>SEJONG</th>
      <th>SEOUL</th>
      <th>ULSAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1994043 rows × 16 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

## HEAR_LEFT
왼쪽귀에 이상이 있는 사람은 당뇨병 유병률이 높으므로 결측치(357)를 당뇨병환자를 기준으로 처리한다.
* 당뇨환자 -> DEAF
* 정상인 -> Normal


```python
mapper = {1: 'Normal', 2 : 'DEAF'}
df['HEAR_LEFT'].replace(mapper, inplace=True)
df[["HEAR_LEFT", "BLDS"]].groupby(['HEAR_LEFT'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HEAR_LEFT</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEAF</td>
      <td>0.631603</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>0.453704</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['HEAR_LEFT'] = df['HEAR_LEFT'].astype('category')
pd.get_dummies(df['HEAR_LEFT'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1994043 rows × 1 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

## HEAR_RIGHT
왼쪽귀에 이상이 있는 사람은 당뇨병 유병률이 높으므로 결측치(355)를 당뇨병환자를 기준으로 처리한다.
* 당뇨환자 -> DEAF
* 정상인 -> Normal


```python
mapper = {1: 'Normal', 2 : 'DEAF'}
df['HEAR_RIGHT'].replace(mapper, inplace=True)
df[["HEAR_RIGHT", "BLDS"]].groupby(['HEAR_RIGHT'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HEAR_RIGHT</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEAF</td>
      <td>0.632988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>0.453874</td>
    </tr>
  </tbody>
</table>
</div>




```python
normal_case = (df['BLDS'] == 1) & (df['HEAR_RIGHT'].isnull()) # fill 0
deaf_case = (df['BLDS'] == 0) & (df['HEAR_RIGHT'].isnull()) # fill 1
df.loc[normal_case,'HEAR_RIGHT'] = df.loc[normal_case,'HEAR_RIGHT'].fillna('Normal')
df.loc[deaf_case,'HEAR_RIGHT'] = df.loc[deaf_case,'HEAR_RIGHT'].fillna('DEAF')
```


```python
df['HEAR_RIGHT'] = df['HEAR_RIGHT'].astype('category')
pd.get_dummies(df['HEAR_RIGHT'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1994043 rows × 1 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

## SMK_STAT_TYPE_CD
흡연자는 당뇨병 유병률이 높으므로 결측치(377)를 당뇨병환자를 기준으로 처리한다.
* 당뇨환자 -> Yes
* 정상인 -> No


```python
mapper = {1: 'No', 2 : 'No', 3: 'Yes'}
df['SMK_STAT_TYPE_CD'].replace(mapper, inplace=True)
df[["SMK_STAT_TYPE_CD", "BLDS"]].groupby(['SMK_STAT_TYPE_CD'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMK_STAT_TYPE_CD</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>0.523379</td>
    </tr>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>0.441921</td>
    </tr>
  </tbody>
</table>
</div>




```python
smoke = (df['BLDS'] == 1) & (df['SMK_STAT_TYPE_CD'].isnull()) # fill 0
non_smoke = (df['BLDS'] == 0) & (df['SMK_STAT_TYPE_CD'].isnull()) # fill 1
df.loc[smoke,'SMK_STAT_TYPE_CD'] = df.loc[smoke,'SMK_STAT_TYPE_CD'].fillna('Yes')
df.loc[non_smoke,'SMK_STAT_TYPE_CD'] = df.loc[non_smoke,'SMK_STAT_TYPE_CD'].fillna('No')
```


```python
df['SMK_STAT_TYPE_CD'] = df['SMK_STAT_TYPE_CD'].astype('category')
pd.get_dummies(df['SMK_STAT_TYPE_CD'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>0</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1994043 rows × 1 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

## DRK_YN
음주는 당뇨병을 악화시키지만, 음주 여부는 당뇨 유병률에 큰 차이를 보이지 않기 때문에 결측치(350820) 개의 행은 삭제 시킨다.


```python
mapper = {1: 'Yes', 0 : 'No'}
df['DRK_YN'].replace(mapper, inplace=True)
df[["DRK_YN", "BLDS"]].groupby(['DRK_YN'], as_index=False).mean().sort_values(by='BLDS', ascending=False)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DRK_YN</th>
      <th>BLDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>0.461851</td>
    </tr>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>0.438254</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 결측치 행 삭제
df = df.dropna(subset = ['DRK_YN'], how = 'any', axis=0)
```


```python
df['DRK_YN'] = df['DRK_YN'].astype('category')
pd.get_dummies(df['DRK_YN'], drop_first=True)
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>999993</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999994</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>1</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1643223 rows × 1 columns</p>
</div>




```python
df = pd.get_dummies(df, drop_first=True)
```

# Numercial features
결측치, 이상치를 적절히 전처리해준다.
AGE_GROUP, HEIGHT, WEIGHT, BMI, WAIST, SIGHT_LEFT, SIGHT_RIGHT, OLIG_PROTE_CD, BP_HIGH, BP_LWST, TOT_CHOLE, TRIGLYCERIDE, HDL_CHOLE, LDL_CHOLE, HMG, CREATININE, SGOT_AST, SGPT_ALT, GAMMA_GTP

## AGE_GROUP
Discrete Numercial type  
결측치, 이상치는 없다


```python
plt.figure(figsize=(15,10))
sns.countplot(df['AGE_GROUP'])
plt.title("AGE_GROUP",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_42_0.png)
    


## HEIGHT
Discrete Numercial type  
결측치, 이상치는 없다


```python
plt.figure(figsize=(15,10))
sns.countplot(df['HEIGHT'])
plt.title("HEIGHT",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_44_0.png)
    


## WEIGHT
Discrete Numercial type  
결측치, 이상치는 없다


```python
plt.figure(figsize=(15,10))
sns.countplot(df['WEIGHT'])
plt.title("WEIGHT",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_46_0.png)
    


## BMI(체질량지수)
Continuous Numercial type
```
df['WEIGHT'] / (df['HEIGHT']/100)**2
```
새로운 feature를 생성해준다.


```python
df['BMI'] = df['WEIGHT'] / (df['HEIGHT']/100)**2
```


```python
sns.kdeplot(df.BMI)
plt.title("BMI distribution")
plt.show()
```


![png](/images/nhis_p/output_49_0.png)
    


## PIBW(체질량지수)
Continous Numercial type  
표준 체중과 비교해 얼마나 차이가 나는지 확인하는 지표
* (0, 90%) : 저체중
* (90%, 110%) : 정상체중
* (110%, 120%) : 과체중
* (120%,~) : 비만

PIBW = 100 * 측정체중 / 표준체중 



```python
df['PIBW'] = np.where(df['SEX_male'] == 1, (100*df['WEIGHT'])/(22*(df['HEIGHT']/100)**2), (100*df['WEIGHT'])/(21*(df['HEIGHT']/100)**2))
```


```python
sns.kdeplot(df.PIBW)
plt.title("PIBW distribution")
plt.show()
```


![png](/images/nhis_p/output_52_0.png)
    


## WAIST
Continous Numercial type  
결측치(394)와 양측에 이상치가 존재한다.  
허리둘레가 999cm 인사람 58명  
허리둘레가 45cm가 안되는 사람 15명  
허리둘레 45cm면 웬만한 마른 아이돌 허리둘레보다 얇은 길이인데 이보다 훨씬 얇은 사람들이 있다.
> 이상치 및 결측치를 ['HEIGHT', 'WEIGHT', 'BMI', 'PIBW']를 이용해 Linear Regression으로 채워준다.


```python
# 900이상의 허리둘레는 잘못 기입된 값이므로 이상치로 처리한다.
df['WAIST'] = np.where(df['WAIST']>900, np.nan, df['WAIST'])
# 40cm미만의 허리둘레는 잘못 기입된 값이므로 이상치로 처리한다.
df['WAIST'] = np.where(df['WAIST']<40, np.nan, df['WAIST'])
```


```python
sns.kdeplot(df.WAIST)
plt.title("WAIST distribution")
plt.show()
```


![png](/images/nhis_p/output_55_0.png)
    


### BMI, PIBW 와 WAIST의 상관관계
각 0.8, 0.75로 높은 상관성을 띈다.


```python
fig, (ax1,ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
sns.scatterplot(data=df, y="BMI", x = 'WAIST', ax=ax1)
sns.scatterplot(data=df, y="PIBW", x = 'WAIST', ax=ax2)
print(df[['PIBW', 'BMI','WAIST']].corr())
```

               PIBW       BMI     WAIST
    PIBW   1.000000  0.987789  0.756360
    BMI    0.987789  1.000000  0.807472
    WAIST  0.756360  0.807472  1.000000




![png](/images/nhis_p/output_57_1.png)
    


### Linear Regression으로 WAIST 결측치 및 이상치 채우기
72% 정도의 정확성을 가진 Linear Regression 모델로 WAIST결측치를 채운 새로운 WAIST_LR feature를 생성하고 기존 WAIST는 삭제


```python
df_copy = df[df['WAIST'].notnull()][['BMI', 'PIBW', 'WAIST', 'HEIGHT', 'WEIGHT']]
X_train, X_test, y_train, y_test = train_test_split(df_copy[['BMI', 'PIBW', 'HEIGHT', 'WEIGHT']], 
                                                    df_copy['WAIST'], random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```




    0.7271066870154038




```python
df['WAIST_pred'] = lr.predict(df[['BMI','PIBW', 'HEIGHT', 'WEIGHT']])
df['WAIST'] = np.where(df['WAIST']>0, df['WAIST'], df['WAIST_pred'])
```


```python
sns.kdeplot(df.WAIST)
plt.title("WAIST distribution")
plt.show()
```


![png](/images/nhis_p/output_61_0.png)
    



```python
# 임시 feature WAIST_pred drop
drop_cols = ['WAIST_pred']
df.drop(drop_cols, axis = 1, inplace = True)
```

## SIGHT_LEFT
Discrete Numercial type  
이상치는 없지만 실명(9.9)을 0으로 바꾸어준다.  
결측치(273)은 수가 적으니 평균시력(1)로 채워준다.  


```python
# 실명 9.9 -> 0
df['SIGHT_LEFT'] = np.where(df['SIGHT_LEFT']>2.5, 0, df['SIGHT_LEFT'])
```


```python
plt.figure(figsize=(15,10))
sns.countplot(df['SIGHT_LEFT'])
plt.title("SIGHT_LEFT",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_65_0.png)
    



```python
df['SIGHT_LEFT'].fillna(1, inplace = True)
```

## SIGHT_RIGHT
Discrete Numercial type  
이상치는 없지만 실명(9.9)을 0으로 바꾸어준다.  
결측치(283)은 수가 적으니 평균시력(1)로 채워준다.  


```python
# 실명 9.9 -> 0
df['SIGHT_RIGHT'] = np.where(df['SIGHT_RIGHT']>2.5, 0, df['SIGHT_RIGHT'])
```


```python
plt.figure(figsize=(15,10))
sns.countplot(df['SIGHT_RIGHT'])
plt.title("SIGHT_RIGHT",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_69_0.png)
    



```python
df['SIGHT_RIGHT'].fillna(1, inplace = True)
```

## OLIG_PROTE_CD
Discrete Numercial type  
결측치(7009)  
데이터상으로 요단백수치와 당뇨는 관계가 없다. 
* 요단백수치 2이상인 사람중 당뇨 유병률 : 5%

멱분포를 따르는것같다. 압도적으로 1이 많기 때문에 결측치 7009개를 1로 채워준다


```python
plt.figure(figsize=(15,10))
sns.countplot(df['OLIG_PROTE_CD'])
plt.title("OLIG_PROTE_CD",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_72_0.png)
    



```python
df['OLIG_PROTE_CD'].fillna(1, inplace = True)
```

## BP_HIGH

Continous Numercial type
결측치(25)수가 적으므로 drop 해준다.  
수축기 혈압이 10의 배수인 부근에 값이 몰려있는것을 발견할 수 있는데 이는 혈압 측정자가 반올림을 하여 10단위 구간에 값들이 몰려있는것 같다.
* 0 ~ 99 : 저혈압(-1)
* 100~120 : 정상(0)
* 121~139 : 전고혈압(1)
* 140~159 : 1단계 고혈압(2)
* 160 ~   : 2단계 고혈압(3)

으로 맵핑해준다.


```python
df = df.dropna(subset = ['BP_HIGH'], how = 'any', axis=0)
```


```python
sns.distplot(df.BP_HIGH)
plt.title("BP_HIGH")
plt.show()
```


![png](/images/nhis_p/output_76_0.png)
    



```python
df['BP_HIGH_level'] = np.where(160 <= df['BP_HIGH'], 3, df['BP_HIGH'])
df['BP_HIGH_level'] = np.where((140<=df['BP_HIGH']) & (df['BP_HIGH']<160), 2, df['BP_HIGH_level'])
df['BP_HIGH_level'] = np.where((121<=df['BP_HIGH']) & (df['BP_HIGH']<140), 1, df['BP_HIGH_level'])
df['BP_HIGH_level'] = np.where((100<=df['BP_HIGH']) & (df['BP_HIGH']<=120), 0, df['BP_HIGH_level'])
df['BP_HIGH_level'] = np.where((df['BP_HIGH'].min()<=df['BP_HIGH']) & (df['BP_HIGH']<100), -1, df['BP_HIGH_level'])
```


```python
plt.figure(figsize=(15,10))
sns.countplot(df['BP_HIGH_level'])
plt.title("BP_HIGH_level",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_78_0.png)
    


## BP_LWST
Continous Numercial type
결측치(1)수가 적으므로 drop 해준다.  
이완기 혈압이 10의 배수인 부근에 값이 몰려있는것을 발견할 수 있는데 이는 혈압 측정자가 반올림을 하여 10단위 구간에 값들이 몰려있는것 같다.
* 0 ~ 59 : 저혈압(-1)
* 60~80 : 정상(0)
* 81~89 : 전고혈압(1)
* 90~99 : 1단계 고혈압(2)
* 100 ~   : 2단계 고혈압(3)

으로 맵핑해준다.


```python
df = df.dropna(subset = ['BP_LWST'], how = 'any', axis=0)
```


```python
sns.distplot(df.BP_LWST)
plt.title("BP_LWST")
plt.show()
```


![png](/images/nhis_p/output_81_0.png)
    



```python
df['BP_LWST_level'] = np.where(100 <= df['BP_LWST'], 3, df['BP_LWST'])
df['BP_LWST_level'] = np.where((90<=df['BP_LWST']) & (df['BP_LWST']<100), 2, df['BP_LWST_level'])
df['BP_LWST_level'] = np.where((81<=df['BP_LWST']) & (df['BP_LWST']<90), 1, df['BP_LWST_level'])
df['BP_LWST_level'] = np.where((60<=df['BP_LWST']) & (df['BP_LWST']<=80), 0, df['BP_LWST_level'])
df['BP_LWST_level'] = np.where((df['BP_LWST'].min()<=df['BP_LWST']) & (df['BP_LWST']<60), -1, df['BP_LWST_level'])
```


```python
plt.figure(figsize=(15,10))
sns.countplot(df['BP_LWST_level'])
plt.title("BP_LWST_level",fontsize=15)
plt.show()
```


![png](/images/nhis_p/output_83_0.png)
    


## HMG
Continous Numercial type  
결측치(24)수가 적으므로 drop 해준다.  
정규분포를 띄지만 약간의 이상치가 존재한다.


```python
df = df.dropna(subset = ['HMG'], how = 'any', axis=0)
```


```python
sns.distplot(df.HMG)
plt.title("HMG")
plt.show()
```


![png](/images/nhis_p/output_86_0.png)
    



```python
sns.boxplot(data=df, x='HMG')
```




    <AxesSubplot:xlabel='HMG'>




![png](/images/nhis_p/output_87_1.png)
    


# 이상치처리가 필요한 features
* CREATININE
* SGOT_AST
* SGPT_ALT
* GAMMA_GTP

4개의 feature들은 모두 공통적으로 극단적인 큰값들이 분포해있는것으로 보아 잘못기입되거나, 오류가아닌 실제로 건강이 안좋아서 나온 수치들로 보여진다. 하지만 이렇게 극단적인 값들을 그대로 가지고 training 시켰을 때, 결과물이 outlier에 의해 치우져질 가능성이 매우크기 때문에 outlier처리를 해야한다.


우선 4개의 feature 결측치의 수는 적으므로 drop 해준다  
4개의 feature의 outlier 분포를 확인한 결과, outlier들은 특정 사람에게 몰려있는 것이아닌 random하게 분포되어있음을 확인했다.


```python
# 4개의 feature 결측치 제거
df = df.dropna(subset = ['CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP'], how = 'any', axis=0)
```


```python
outlier = df[['CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP']].copy()
for col in outlier :
  q1 = outlier[col].quantile(0.25)
  q3 = outlier[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  print('{}\'s upper bound : {}, lower bount : {}'.format(col, round(upper_bound, 2), round(lower_bound, 2)))
  outlier[col] = np.where(outlier[col] > upper_bound, np.nan, outlier[col])
  outlier[col] = np.where(outlier[col] < lower_bound, np.nan, outlier[col])
msno.matrix(outlier)
plt.show()

outlier.isnull().sum()
print(outlier.isnull().sum())
print('-'*30)
print('전체 이상치 개수 :', len(outlier) - len(outlier.dropna()))
```

    CREATININE's upper bound : 1.45, lower bount : 0.25
    SGOT_AST's upper bound : 44.0, lower bount : 4.0
    SGPT_ALT's upper bound : 52.5, lower bount : -7.5
    GAMMA_GTP's upper bound : 81.0, lower bount : -23.0




![png](/images/nhis_p/output_91_1.png)
    


    CREATININE     12069
    SGOT_AST       95285
    SGPT_ALT      119253
    GAMMA_GTP     149083
    dtype: int64
    ------------------------------
    전체 이상치 개수 : 250811


> 1행 : 기존 데이터 분포  
> 2행 : 이상치 제거한 데이터 분포  
> 3행 : 이상치 제거한 boxplot


```python
fig, axes = plt.subplots(nrows = 3, ncols=4, figsize=(18, 12))
for i, col in enumerate(outlier) :
  sns.kdeplot(df[col], ax = axes[0][i])
  sns.countplot(outlier[col],ax = axes[1][i])
  sns.boxplot(data=outlier, x=col, orient="v", ax = axes[2][i])
plt.setp(axes.flat, xlabel=None, ylabel=None)
pad = 5
rows = ['include outlier', 'exclude outlier', 'exclude outlier boxplot']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.tight_layout()
plt.show()
```


![png](/images/nhis_p/output_93_0.png)
    



```python
outlier = ['CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP']

for col in outlier :
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  df[col] = np.where(df[col] > upper_bound, np.nan, df[col])
  df[col] = np.where(df[col] < lower_bound, np.nan, df[col])

df = df.dropna(subset = outlier, how = 'any', axis=0)
```

## 중간점검
결측치가 가장많이 몰려있는 콜레스테롤 관련 features들
* TOT_CHOLE
* TRIGLYCERIDE
* HDL_CHOLE
* LDL_CHOLE

을 제외한 데이터 전처리를 끝냈다.


```python
msno.matrix(df)
plt.show()
print(df.isnull().sum())
```


![png](/images/nhis_p/output_96_0.png)
    


    AGE_GROUP                    0
    HEIGHT                       0
    WEIGHT                       0
    WAIST                        0
    SIGHT_LEFT                   0
    SIGHT_RIGHT                  0
    BP_HIGH                      0
    BP_LWST                      0
    BLDS                         0
    TOT_CHOLE               369826
    TRIGLYCERIDE            369832
    HDL_CHOLE               369831
    LDL_CHOLE               374327
    HMG                          0
    OLIG_PROTE_CD                0
    CREATININE                   0
    SGOT_AST                     0
    SGPT_ALT                     0
    GAMMA_GTP                    0
    SEX_male                     0
    SIDO_CB                      0
    SIDO_CN                      0
    SIDO_DAGU                    0
    SIDO_DAJEON                  0
    SIDO_FOREIGN                 0
    SIDO_GANGWON                 0
    SIDO_GB                      0
    SIDO_GN                      0
    SIDO_GYEONGGI                0
    SIDO_INCHEON                 0
    SIDO_JB                      0
    SIDO_JN                      0
    SIDO_KWANGJU                 0
    SIDO_SEJONG                  0
    SIDO_SEOUL                   0
    SIDO_ULSAN                   0
    HEAR_LEFT_Normal             0
    HEAR_RIGHT_Normal            0
    SMK_STAT_TYPE_CD_Yes         0
    DRK_YN_Yes                   0
    BMI                          0
    PIBW                         0
    BP_HIGH_level                0
    BP_LWST_level                0
    dtype: int64


# CHOLE Data
* TOT_CHOLE
* TRIGLYCERIDE
* HDL_CHOLE
* LDL_CHOLE


```python
outlier = df[['TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE']].copy()
a = len(outlier.dropna())
for col in outlier :
  q1 = outlier[col].quantile(0.25)
  q3 = outlier[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  print('{}\'s upper bound : {}, lower bount : {}'.format(col, round(upper_bound, 2), round(lower_bound, 2)))
  outlier[col] = np.where(outlier[col] > upper_bound, np.nan, outlier[col])
  outlier[col] = np.where(outlier[col] < lower_bound, np.nan, outlier[col])
msno.matrix(outlier)
plt.show()

outlier.isnull().sum()
print(outlier.isnull().sum())
print('-'*30)
print('전체 이상치 및 outlier 개수 :', len(outlier) - len(outlier.dropna()))
```

    TOT_CHOLE's upper bound : 291.5, lower bount : 95.5
    TRIGLYCERIDE's upper bound : 266.0, lower bount : -46.0
    HDL_CHOLE's upper bound : 94.5, lower bount : 18.5
    LDL_CHOLE's upper bound : 200.0, lower bount : 24.0




![png](/images/nhis_p/output_98_1.png)
    


    TOT_CHOLE       381142
    TRIGLYCERIDE    422080
    HDL_CHOLE       388140
    LDL_CHOLE       385875
    dtype: int64
    ------------------------------
    전체 이상치 및 outlier 개수 : 454367



```python
fig, axes = plt.subplots(nrows = 3, ncols=4, figsize=(18, 12))
for i, col in enumerate(outlier) :
  sns.kdeplot(df[col], ax = axes[0][i])
  sns.countplot(outlier[col],ax = axes[1][i])
  sns.boxplot(data=outlier, x=col, orient="v", ax = axes[2][i])
plt.setp(axes.flat, xlabel=None, ylabel=None)

pad = 5
rows = ['include outlier', 'exclude outlier', 'exclude outlier boxplot']
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.tight_layout()
plt.show()
```


![png](/images/nhis_p/output_99_0.png)
    



```python
# 이상치 제거
cols = ['TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE']
for col in cols :
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  df = df[(lower_bound<=df[col]) & (df[col] <= upper_bound) | (df[col].isnull())]
```


```python
countinous = ['AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST', 'SIGHT_LEFT', 
              'SIGHT_RIGHT', 'BP_HIGH', 'BP_LWST', 'BLDS', 'TOT_CHOLE',
              'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'HMG', 'OLIG_PROTE_CD',
              'CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP', 'BMI', 'PIBW', 
              'BP_HIGH_level', 'BP_LWST_level']
corrMatt = df[countinous].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
f,ax = plt.subplots(figsize=(30, 30))
#sns.heatmap(corrMatt, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True, ax = ax)

```




    <AxesSubplot:>




![png](/images/nhis_p/output_101_1.png)
    


## Regression 으로 결측치 채우기
Linear Regression 모델링을 통해 결측치를 채워본다.


```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


data = df.copy()
data.dropna(inplace = True)
y_tot = data['TOT_CHOLE']
y_tri = data['TRIGLYCERIDE']
y_hdl = data['HDL_CHOLE']
y_ldl = data['LDL_CHOLE']

drop_cols = ['TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'BLDS']
data.drop(drop_cols, axis = 1, inplace = True)
```

## Linear Regression Train & Predict
예측할 4개의 feature과 Target을 제외한 나머지 feature들로 Linear Regression을 시행한 결과 MSE, score이 터무니없이 작게 나와서 예측치로 결측치를 채울 수 없다는 결론을 내렸다. 따라서 결측치는 삭제처리한다.


```python
X_train, X_test, y_tot_train, y_tot_test = train_test_split(data, y_tot, test_size = 0.3, random_state=42)  
X_train, X_test, y_tri_train, y_tri_test = train_test_split(data, y_tri, test_size = 0.3, random_state=42)  
X_train, X_test, y_hdl_train, y_hdl_test = train_test_split(data, y_hdl, test_size = 0.3, random_state=42)  
X_train, X_test, y_ldl_train, y_ldl_test = train_test_split(data, y_ldl, test_size = 0.3, random_state=42)  

sc = StandardScaler().fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)
```


```python
model_tot = LinearRegression()
model_tot.fit(X_train_sc, y_tot_train)
print("TOT score : ",model_tot.score(X_test_sc, y_tot_test))
y_pred = model_tot.predict(X_test_sc) 
print("TOT MSE : ", sum((y_tot_test - y_pred)**2) / len(y_tot_test))
```

    TOT score :  0.05576215500393267
    TOT MSE :  1113.3989897779566



```python
model_tri = LinearRegression()
model_tri.fit(X_train_sc, y_tri_train)
print("TRI score : ",model_tri.score(X_test_sc, y_tri_test))
y_pred = model_tri.predict(X_test_sc) 
print("TRI MSE : ", sum((y_tri_test - y_pred)**2) / len(y_tri_test))
```

    TRI score :  0.20016898079679968
    TRI MSE :  2092.9693749041735



```python
model_hdl = LinearRegression()
model_hdl.fit(X_train_sc, y_hdl_train)
print("HDL score : ",model_hdl.score(X_test_sc, y_hdl_test))
y_pred = model_hdl.predict(X_test_sc) 
print("HDL MSE : ", sum((y_hdl_test - y_pred)**2) / len(y_hdl_test))
```

    HDL score :  0.1983517184445205
    HDL MSE :  143.9153872584845



```python
model_ldl = LinearRegression()
model_ldl.fit(X_train_sc, y_ldl_train)
print("LDL score : ",model_ldl.score(X_test_sc, y_ldl_test))
y_pred = model_ldl.predict(X_test_sc) 
print("LDL MSE : ", sum((y_ldl_test - y_pred)**2) / len(y_ldl_test))
```

    LDL score :  0.041472068147774266
    LDL MSE :  949.2267775426899



```python
df.dropna(inplace = True)
msno.matrix(df)
plt.show()
```


​    
![png](/images/nhis_p/output_110_0.png)
​    



```python
df.describe()
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE_GROUP</th>
      <th>HEIGHT</th>
      <th>WEIGHT</th>
      <th>WAIST</th>
      <th>SIGHT_LEFT</th>
      <th>SIGHT_RIGHT</th>
      <th>BP_HIGH</th>
      <th>BP_LWST</th>
      <th>BLDS</th>
      <th>TOT_CHOLE</th>
      <th>TRIGLYCERIDE</th>
      <th>HDL_CHOLE</th>
      <th>LDL_CHOLE</th>
      <th>HMG</th>
      <th>OLIG_PROTE_CD</th>
      <th>CREATININE</th>
      <th>SGOT_AST</th>
      <th>SGPT_ALT</th>
      <th>GAMMA_GTP</th>
      <th>SEX_male</th>
      <th>SIDO_CB</th>
      <th>SIDO_CN</th>
      <th>SIDO_DAGU</th>
      <th>SIDO_DAJEON</th>
      <th>SIDO_FOREIGN</th>
      <th>SIDO_GANGWON</th>
      <th>SIDO_GB</th>
      <th>SIDO_GN</th>
      <th>SIDO_GYEONGGI</th>
      <th>SIDO_INCHEON</th>
      <th>SIDO_JB</th>
      <th>SIDO_JN</th>
      <th>SIDO_KWANGJU</th>
      <th>SIDO_SEJONG</th>
      <th>SIDO_SEOUL</th>
      <th>SIDO_ULSAN</th>
      <th>HEAR_LEFT_Normal</th>
      <th>HEAR_RIGHT_Normal</th>
      <th>SMK_STAT_TYPE_CD_Yes</th>
      <th>DRK_YN_Yes</th>
      <th>BMI</th>
      <th>PIBW</th>
      <th>BP_HIGH_level</th>
      <th>BP_LWST_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
      <td>937975.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10.517616</td>
      <td>162.002399</td>
      <td>62.181599</td>
      <td>80.212072</td>
      <td>0.950041</td>
      <td>0.947671</td>
      <td>121.454282</td>
      <td>75.357764</td>
      <td>0.403228</td>
      <td>191.829651</td>
      <td>109.303008</td>
      <td>57.447467</td>
      <td>112.449188</td>
      <td>14.125549</td>
      <td>1.078551</td>
      <td>0.841988</td>
      <td>22.653508</td>
      <td>20.599382</td>
      <td>25.501897</td>
      <td>0.505338</td>
      <td>0.033069</td>
      <td>0.041271</td>
      <td>0.047271</td>
      <td>0.030269</td>
      <td>0.010303</td>
      <td>0.029905</td>
      <td>0.053449</td>
      <td>0.066947</td>
      <td>0.248925</td>
      <td>0.057672</td>
      <td>0.035787</td>
      <td>0.035373</td>
      <td>0.027682</td>
      <td>0.004433</td>
      <td>0.184352</td>
      <td>0.024929</td>
      <td>0.968739</td>
      <td>0.969776</td>
      <td>0.194926</td>
      <td>0.551999</td>
      <td>23.588539</td>
      <td>109.680289</td>
      <td>0.533975</td>
      <td>0.303479</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.837100</td>
      <td>9.229756</td>
      <td>11.750765</td>
      <td>9.191787</td>
      <td>0.344178</td>
      <td>0.342207</td>
      <td>14.283575</td>
      <td>9.665989</td>
      <td>0.600539</td>
      <td>34.335348</td>
      <td>51.149954</td>
      <td>13.406158</td>
      <td>31.477233</td>
      <td>1.552392</td>
      <td>0.385756</td>
      <td>0.190751</td>
      <td>6.076824</td>
      <td>9.101093</td>
      <td>14.839650</td>
      <td>0.499972</td>
      <td>0.178817</td>
      <td>0.198916</td>
      <td>0.212218</td>
      <td>0.171328</td>
      <td>0.100980</td>
      <td>0.170325</td>
      <td>0.224928</td>
      <td>0.249931</td>
      <td>0.432390</td>
      <td>0.233122</td>
      <td>0.185758</td>
      <td>0.184721</td>
      <td>0.164060</td>
      <td>0.066433</td>
      <td>0.387772</td>
      <td>0.155910</td>
      <td>0.174022</td>
      <td>0.171202</td>
      <td>0.396144</td>
      <td>0.497289</td>
      <td>3.339794</td>
      <td>15.326171</td>
      <td>0.750506</td>
      <td>0.685260</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>130.000000</td>
      <td>25.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>63.000000</td>
      <td>31.000000</td>
      <td>0.000000</td>
      <td>96.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>24.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.300000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.486993</td>
      <td>59.461870</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>155.000000</td>
      <td>55.000000</td>
      <td>74.000000</td>
      <td>0.700000</td>
      <td>0.700000</td>
      <td>110.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>168.000000</td>
      <td>70.000000</td>
      <td>48.000000</td>
      <td>90.000000</td>
      <td>13.100000</td>
      <td>1.000000</td>
      <td>0.700000</td>
      <td>18.000000</td>
      <td>14.000000</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.403092</td>
      <td>99.103117</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.000000</td>
      <td>160.000000</td>
      <td>60.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>120.000000</td>
      <td>75.000000</td>
      <td>0.000000</td>
      <td>191.000000</td>
      <td>98.000000</td>
      <td>56.000000</td>
      <td>111.000000</td>
      <td>14.100000</td>
      <td>1.000000</td>
      <td>0.800000</td>
      <td>22.000000</td>
      <td>19.000000</td>
      <td>21.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>23.437500</td>
      <td>109.013428</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.000000</td>
      <td>170.000000</td>
      <td>70.000000</td>
      <td>86.000000</td>
      <td>1.200000</td>
      <td>1.200000</td>
      <td>130.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>215.000000</td>
      <td>139.000000</td>
      <td>66.000000</td>
      <td>134.000000</td>
      <td>15.300000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>25.000000</td>
      <td>32.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>25.711662</td>
      <td>118.738404</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18.000000</td>
      <td>190.000000</td>
      <td>140.000000</td>
      <td>145.000000</td>
      <td>2.500000</td>
      <td>2.500000</td>
      <td>273.000000</td>
      <td>185.000000</td>
      <td>2.000000</td>
      <td>291.000000</td>
      <td>263.000000</td>
      <td>95.000000</td>
      <td>200.000000</td>
      <td>25.000000</td>
      <td>6.000000</td>
      <td>1.400000</td>
      <td>44.000000</td>
      <td>52.000000</td>
      <td>81.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>45.714286</td>
      <td>209.891119</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (937975, 46)



# 이상치, 결측치 처리완료
데이터 변화 : `2000000 x 34` -> `937975 x 46`

# Feature Create


## ABD_FAT
복부비만 수치이다.  
허리둘레가 남자 90, 여자 85이상이면 복부비만(1), 정상(0)으로 해준다.


```python
fat = (df['WAIST'] >= 90) & (df['SEX_male'] == 1) | (df['WAIST'] >= 85) & (df['SEX_male'] == 0)
df['ABD_FAT'] = np.where(fat, 1, 0)
```

## GAMMA_GTP_level
* `2 if GAMMA_GTP >= 200` 위험, but **GAMMA_GTP.max = 81**
* `1 if (GAMMA_GTP > 63 and SEX_male == 1) or (GAMMA_GTP > 35 and SEX_male == 0)` 경고
* `0` 정상


```python
warn = (df['GAMMA_GTP'] > 63) & (df['SEX_male'] == 1) | (df['GAMMA_GTP'] > 35) & (df['SEX_male'] == 0)
df['GAMMA_GTP_level'] = np.where(warn, 1, 0)
```
