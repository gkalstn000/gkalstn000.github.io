---
layout: post
title:  "개빡치는 정밀도 줄이기 & LoRA"
categories: 실습
date: 2025-11-22 11:40:18 +0900
tags: AI Mixed-Precision 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

![fp8 precision 창나는 현상](/images/ai_techs/fp8_씨발년.png)
검은 영역 생성하는 모델 학습에서 train 환경 `bfloat16` 에서는 문제없던게 테스트환경 `fp8`로 생성하니 테두리 부분이 개박살나는 현상. 이건 아직도 해결방법 못찾음ㅅㅂ (거의 한달넘게 실험바꿔가며 헉습/실험 했는데 해결 못함ㅋ).  
* 근데 원인은 몰라도 우회할 방법을 고안해 냈는데 이직을 하게되어 가설이 맞는지는 당분간 확인을 할 수 없게 되었음...

이게 아무래도 높은 precision에서 낮은 precision으로 내려가면서 발생하는 문제가 확실해 보이는데, 근데 하필이면 딱 경계면에 저렇게 폭격이 가해지는 원인이 뭘까?  
이걸 파악하기 위해서는 precision 낮추는 과정을 좀 low-level 에서 파악할 필요성이 느껴진다.  

---







그 전에 굉장히 흥미로운 현상을 하나 발견했다.

![casting.png](/images/ai_techs/casting.png)
내가 직장생활 하면서 정말 많이 공부한 오픈소스 프로젝트인 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 소스코드를 보자. 여기에서는 `fp8`이든 `float32`든 타입 변경할 때나 혹은 device 바꿀 때나 `to(dtype=torch.float32)` 그냥 이렇게 쓰는 법이 없고 무조건 지들이 만든 함수로 캐스팅을 한다.
* 심지어 회사 프사가 애니 프사임 ㄷㄷ...

그림을 보면 원본 `weight`를 캐스팅 하는데 `ComfyUI` 방식과 그냥 대충 때려박는 `to` 방식의 결과물에서 이상한 차이가 보인다. 전체적으로 틀려지는게 아니라 특정 element들만 조금씩 다르다. 물론 생성 결과물은 육안으로 확인이 불가능할 정도로 거의 없다(진짜 쪼금 있음).  
`ComfyUI` 회사에서 근무하는 ㅈ고수들이 단순히 `to`로 때려박는것이 위험하다고 판단하여 지들만의 방식으로 캐스팅을 하는 것이겠지만 내가 알아보고 싶은 것은 3가지이다.
* `fp8`은 도대체 뭘까. 어떤식으로 줄이는거임.
* `ComfyUI`에서 쓰는 `stochastic_rounding`과 `to`는 차이가 뭘까? 
* `bfloat16`을 `fp8`로 바꿨을 때 왜 하필이면 테두리 부분만 박살이날까? 타입 캐스팅할 때 그 부분을 완화시킬 순 없을까?
  * 테두리 말고 내부가 박살나는게 차라리 나은데...

만약 이 문제를 해결한다면 내년에 유력한 발롱도르 후보자가 될 수 있다는 강한 확신이 든다. 우선 하나씩 파헤쳐 본다.

## fp8 은 뭐하는놈이고 `e4m3fn` 같이 뒤에붙은 거슬리는 이거는 뭐임?
나는 개발자가 아니니 쓸일 없을꺼다 하고 한귀로 흘렸던 **부동소수점** 에 대한 공부가 선행되어야했다. 

![부동소수점](/images/ai_techs/floating.png)

공부를 하고 나니 왜 컴퓨터가 표현할 수 없는 숫자가 존재하는지 그런 것들이 이해가 되었고, fp8방식 특히 `e5m2` 이거는 검정고무신 단칸방에 8명 모여사는 그 집구석도 아니고 가수부가 이래 작아도 되는가 하는 생각마저 들게 했다.  

8비트는 실수를 표현하기에 턱없이 공간이 부족하지만 그럼에도 불구하고 거대모델의 병목 및 크기 해결하기 위해 학습/추론에 사용된다.  
병목의 원인은 '계산' 이 아닌 '메모리 이동'에서 발생한다.
* FP32 (4byte): 너무 무거워서 GPU 메모리에서 가져오는데 한세월 걸림.
* FP8 (1byte): 4배 더 많은 데이터를 실어 나를 수 있음.

---

### `E4M3` vs `E5M2`

| FP8-E4M3 (정밀도 중시)                           | FP8-E5M2 (범위 중시)                                        | 
|---------------------------------------------|---------------------------------------------------------|
| 부호(1) + 지수(4) + 가수(3)                       | 부호(1) + 지수(5) + 가수(2)                                   | 
| 가중치(Weight), 활성화값(Activation)               | 기울기(Gradient)                                      |
| 숫자의 디테일을 잘 살림, 그러나 지수칸이 4칸이라 최대 448까지 표현가능. | 지수가 5칸(FP16과 동일 수준)이라 아주 큰 수, 아주 작은 수를 커버 하지만 숫자가 듬성듬성함 |


### 실전 연산 메커니즘: Mixed Precision
FP8이 실제로 GPU(H100 등)에서 돌아가는 방식은 **"입력은 가볍게, 계산은 무겁게"** 이다.
* 실 계산시 더 높은 칸으로 올려 계산한 뒤 다시 fp8로 복귀 
  * `12,300 + 1 = 12,300` 이런 경우를 막기 위해

#### step 1
* FP32 상태인 데이터를 FP8로 캐스팅
* 이때 그냥 자르면 정보가 다 날아가니, 데이터 분포에 맞춰 **스케일링(Scaling)** 진행
  * 숫자들을 FP8 범위 안으로 욱여넣음

#### step 2
* 텐서 코어 내부에서 $A \times B$ 곱셈을 수행
* 8비트 곱셈은 회로가 단순해서 개빠름

#### step 3
* 곱한 값들을 더할 때는 FP32 (또는 FP16) 으로 변환해서 계산
* 가수 방이 모자라서 1을 더해도 무시당하는 현상 없애기 위해

#### step 4
* 합쳐진 결과(FP32)를 다음 레이어로 보내기 위해 다시 FP8로 변환

그럼 여기서 계산 끝나고 내보낼 때 정밀도 줄이느라 숫자 날라가는건 인정하는 부분이지만, 어떻게 값을 날려야 검정 테두리 영역을 자연스럽게 할 수 있을까가 문제임.  
단순 올림? 내림? 반올림? 

## ComfyUI에서 쓰는 변환 방식은 뭐하는걸까
아래의 함수들은 `ComfyUI`에서 사용하는 `fp8` 변환 함수다. 코드가 살짝 복잡해서 좀 뜯어봐야하겠는데 뭔진 몰라도 이렇게 정성스럽게 해줌에도 테두리 박살나는걸 해결 못하는 상황이다.  
그렇다면 그냥 `weight.to(dtype=torch.fp8)` 변환 방식과 아래의 방식은 어떤 차이를 낼까.

```python
def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign

def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        return output

    return value.to(dtype=dtype)

```

일반적으로 쓰는 `weight.to(dtype=torch.float8)`은 가장 가까운 값으로 반올림을 하는 방식이다.  
이 방식의 큰 문제점은 만약 0.7인 값이 100만개 있다면 전부 1.0으로 쏠리는 bias 현상이 발생한다. 원래 평균 0.7이던게 1.0으로 변하는 아주 큰 차이를 만들어내버림.  

`ComfyUI` 햄들은 확률적 반올림을 통해 평균을 크게 왜곡하지 않도록 제어를 하는 철학을 가지고 위와같은 설계를 했음.
* 논리: "0.7이니까 70% 확률로 1, 30% 확률로 0으로 보내자."


### `stochastic_rounding` 회뜨기

우선 코드를 보면 확률 어쩌고 답게 random seed를 만들어내는 부분을 볼 수 있다.
```python
def stochastic_rounding(value, dtype, seed=0):
    ...
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    
    ...
    return output

```

확률적으로 올리고 버리지만 같은 입력에 대해서는 동일 output을 유지하기 위한 장치임.  
레이어 이름이나 데이터를 기반으로 고정된 seed값을 사용한다.  

---

**VRAM 피크(Peak) 관리**  
```python
output = torch.empty_like(value, dtype=dtype)
output._copy(~~)
```

이거 늘 의문이었음. `ComfyUI`에서는 device 바꿀 때, dtype 바꿀 때 `.to(device=~~, dtype=~~)`안쓰고 항상 empty_like 만들고난 다음에 여기다가 `_copy`로 집어넣음.  
이 방식은 중간 계산과정에서 잡아먹히는 메모리 비용을 획기적으로 줄이는, 마치 카드할부 결제 같은 방식임.  

예를 들어서 2GB 짜리 텐서 `dtype` 변환해서 1GB 짜리 텐서 만들 때 발생하는 계산 과정을보면
```python
exponent = ...      # 2GB (임시 메모리 1)
mantissa = ...      # 2GB (임시 메모리 2)
noise = torch.rand(...) # 2GB (임시 메모리 3)
result = ...        # 1GB (최종 결과 FP8)
```
원본(2GB) + 임시변수들(6GB 이상) + 결과(1GB) $\approx$ 9GB가 소모되는 현상이 발생함.

```python
def stochastic_rounding(value, dtype, seed=0):
    ...
    
    output = torch.empty_like(value, dtype=dtype)
    num_slices = max(1, (value.numel() / (4096 * 4096)))
    slice_size = max(1, round(value.shape[0] / num_slices))
    for i in range(0, value.shape[0], slice_size):
        output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
    return output

```

`ComfyUI`에서는 데이터를 4096 * 4096 (약 1,600만 개) 단위 (대략 64MB)로 잘라내고 변환을 반복함. 이렇게 하면 순간 최대 메모리는 고작 300MB 정도밖에 안치솟음.  
즉, 나같이 8GB, 12GB 급 게임용 GPU 가진 사람들도 메모리 안터뜨리면서 `ComfyUI`로 이미지 생성 할 수 있도록 해주는 기술임.  

`_copy(~~~)` 를 쓰는 이유또한 마찬가지. `torch.empty_like(value, dtype=dtype)`를 통해 딱 필요한 만큼의 메모리 미리 잡아두고, 이미 계산 완료 된 값을 `_copy`를 통해 넣기 때문에 메모리 재할당, 이동이 전혀 발생하지 않게됨.
* stack, cat 방식은 메모리 추가 사용됨.

---

**`manual_stochastic_round_to_float8`**
fp8의 스펙(E4M3, E5M2)에 맞춰서 부호, 지수, 가수를 각각 계산하고 마지막에 합체하는 함수임.  
이 코드는 많이 빡새서 gemini 주석 그대로 활용.
```python
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    # 1. 설정값 세팅 (FP8 종류에 따라 비트 수와 바이어스 결정)
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    
    # 2. 전처리
    x = x.half() # FP16으로 변환 (계산 효율)
    sign = torch.sign(x) # 부호(+/-) 저장해둠
    abs_x = x.abs()
    
    # 3. 지수(Exponent) 계산
    # log2를 취하면 지수(2의 몇 승인지)가 나옴.
    # 바이어스(Bias)를 더해주고, 비트 범위(0 ~ 2^4-1)를 넘지 않게 clamp(가위질)함.
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # 4. 정규수/비정규수 구분 (normal_mask)
    # 지수가 0이면 "너무 작은 숫자"라서 특별 취급해야 함.
    normal_mask = ~(exponent == 0)

    # 5. 가수(Mantissa) 계산 (위에서 분석한 함수 호출)
    # ============  여기서 확률적으로 반올림해 bias 최소화함 ================
    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, ...)
    # ============  여기서 확률적으로 반올림해 bias 최소화함 ================

    # 6. 최종 재조립 (Reconstruction)
    # 공식: (-1)^sign * 2^(exponent-bias) * (1 + mantissa)
    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x), # 정규수: 1.xxx 복구
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x                # 비정규수: 0.xxx 복구
    )

    # 7. 안전장치 (Clamping)
    # 확률적 반올림을 하다 보면 값이 튀어서 FP8 범위를 넘을 수 있음.
    # FP8의 최소/최대값 안으로 강제로 욱여넣음.
    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign
```

이 코드는 수학적 확률론을 이용한 화질 손실 보정 알고리즘으로, `log2, floor, pow` 같은 함수를 써서 상당히 느리지만, 한번만 해두면 되기 때문에 최대한 이미지 품질을 살릴 수 있도록 **장인 정신이 깃든 코드** 라고 한다 (by gemini 극찬).
* 실제로 모델 로드하고 `fp8`로 바꿀 때 거의 20~30초 가까이 걸린다. 

---

**`calc_mantissa`**
```python
def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    # [수식 해부 1] 정규화 (Normalization)
    # 목표: 실수 x를 가수부 정수(0, 1, 2...)로 변환
    mantissa_scaled = torch.where(
        normal_mask,
        # Case A: 일반적인 숫자 (Normalized)
        # 공식: (값 / 2^지수 - 1) * 2^가수비트
        # 1. (abs_x / 2^exponent) -> 1.xxxx 형태가 됨
        # 2. (- 1.0) -> 0.xxxx (앞의 1 제거, 히든 비트 처리)
        # 3. (* 2^MANTISSA_BITS) -> 소수를 정수 범위로 뻥튀기 (예: 0~7)
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        
        # Case B: 아주 작은 숫자 (Denormalized)
        # 0 근처의 숫자는 공식이 다름. 히든 비트 1이 없다고 가정.
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    # [수식 해부 2] 노이즈 주입 (Stochastic Rounding의 핵심!)
    # 정수화된 실수(예: 3.7)에 0~1 사이의 난수를 더함.
    # 3.7 + 0.1(난수) = 3.8 -> floor -> 3 (내림)
    # 3.7 + 0.8(난수) = 4.5 -> floor -> 4 (올림)
    # 결과적으로 0.7의 확률로 4가 되고, 0.3의 확률로 3이 됨.
    mantissa_scaled += torch.rand(..., generator=generator)
    
    # [수식 해부 3] 복원
    # 정수로 만든 값을 다시 소수점 형태로 되돌림 (.floor()로 소수점 날림)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)
```

와 이 코드도 존나 빡쌤. 진짜 어떻게 이런놈이 있는가 생각이 들정도로 정교하게 짰음. ㄹㅇ 벽느껴짐.  
요약
> 실수(3.14)를 가져와서 가수 비트(눈금)에 맞춰 정수(3.7 같은 스케일)로 바꾼 뒤, 랜덤 노이즈를 섞고 확률적으로 소수점을 칼같이 잘라버리는(floor) 과정  

여기서 `normal_mask`는 exponent == 0 인 경우에 0근처의 미세한 가중치들이 0으로 싹 죽는것을 방지하기위한 장치라고 한다. 0으로 죽어버리면 이미지의 어두운 명암 디테일이 뭉개질 수 있다.

## 그래서 왜 테두리만 박살났던걸까
![fp8 precision 창나는 현상](/images/ai_techs/fp8_씨발년.png)

우선 검정색 테두리 부분은 이미지 정보 중 가장 고주파(High Frequency) 영역이다 
* 검정 (0)에서 그림 (0.8) 픽셀값으로 넘어가는 경계.

![frequenct](/images/ai_techs/high_freq.png)


`bfloat16`에서는 [0, 0, 0.6, 0.8] 이런 값을 받아도 레이어를 거치며 고주파 부분이 자연스럽게 완화되어 [0.1, 0.24, 0.43, 0.54, 0.6, 0.71] 이렇게 변하지만, `fp8` 에서는 텐서 중 가장 큰값을 기준으로 버림 눈금을 넓혀버리기 때문에 부드러움이 완화가 잘 되지 않아 [0, 0. 0.3, 0.5, 0.6, 0.7] 이렇게 여전히 고주파가 남아있을 수 있다고 한다.

### 해결책
**1. 검정 영역이 포함된 `condition_image` 처리하는 첫 빠따만 `bfloat16` 그대로 사용하게 하기.**
아 이거는 `ComfyUI unet_loader` 건드려야해서 개빡새 보이긴하는데 가장 일리있는 해결책으로 판단됨. 다행인거는 gemini가 이미 캐스팅 된 fp8을 그냥 bfloat16으로 해주는 거 효과가 아예 없지는 않을꺼라고 함.

**2. 검정 색말고 noise로 채우고 학습하기.**
얘는 뭐 하는거 시간이 걸릴 뿐이지 큰 어려움은 없긴한데 걸리는 점이, timestep 마다 모델 내부에서 다르게 동작하는 diffusion 모델에 보통 condition image는 `t=0`으로 박아넣고 임베딩을 함. 근데 거기에 노이즈가 섞여있으면 괜찮을라나 그게 좀 걸리는데 눈으로 확인 해봐야할듯.

**3. fp8 회피 기동 (Mask Dilation for Inpainting)**
![trick](/images/ai_techs/train_trick.png)
가운데 Model A 는 다른사람이 오픈소스로 공개한 모델 결과물인데, 로션 주둥이를 보면 살짝 깎여서 더 얇아진 모습을 볼 수 있다. 얘가 학습 condition 이미지 만들 때, 마스크를 따놓고 뭔가 장난질을 쳐서 모델이 생성할 때 원본을 깎아 먹는 대신 테두리 영역을 부드럽게 되도록 한것이 틀림 없다.  

![trick](/images/ai_techs/genback_7000_another_case.png)

model A 의 또다른 결과물인데 마스크가 잘못따져서 쿠키 밑에 흰색영역이 남아있는데 감쪽같이 채워버린다.  
아마 내 추측인데 얘는 condition_image를 아래처럼 만든거 같음.
1. 마스크를 따서 배경을 검정으로 만들어 버림.
2. 검정부분을 안쪽으로 3~5픽셀 좁히는데 그 부분은 흰색으로 만듬.
3. 월마리아 마냥 foreground 바깥에 흰색 장벽이 생기고 그 바깥을 검정색으로 채움.

어쨌든 이건 테스트해볼 기회가 생기면 테스트 해봐야겠음ㅇㅇ  

양자화도 같이 파보려했는데 내용이 상당히 길어져서 그건 다음기회에