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
* `fp8`은 도대체 뭘까. 어떤식으로 줄이고, 왜 얘는 덧셈연산이 안되는걸까.
* `ComfyUI`에서 쓰는 `stochastic_rounding`과 `to`는 차이가 뭘까? 
* `bfloat16`을 `fp8`로 바꿨을 때 왜 하필이면 테두리 부분만 박살이날까? 타입 캐스팅할 때 그 부분을 완화시킬 순 없을까?
  * 테두리 말고 내부가 박살나는게 차라리 나은데...

만약 이 문제를 해결한다면 내년에 유력한 발롱도르 후보자가 될 수 있다는 강한 확신이 든다. 우선 하나씩 파헤쳐 본다.

## fp8 은 뭐하는놈이고 `e4m3fn` 같이 뒤에붙은 거슬리는 이거는 뭐임?

## ComfyUI에서 쓰는 변환 방식은 뭐하는걸까

## 그래서 왜 테두리만 박살났던걸까








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