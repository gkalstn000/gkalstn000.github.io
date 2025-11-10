---
layout: post
title:  "Accelerater 테이크 다운 (2편, 분산학습)"
categories: 실습
date: 2025-11-08 11:40:18 +0900
tags: AI Optimizer 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

분산학습이 도대체 뭔지 아직도 사실 잘 모른다.  GPU 여러개에 모델 복사해 놓고 backprop 할 때 gradiend만 취합해서 보내는건지, GPU 여러개를 하나처럼 쓰는건지, 어떻게 쓰는건지 등등.  
이참에 확실히 정리할 예정.

# 분산 학습

| 구분 | 개념 |                       
|------|----|
|Single GPU | 한 GPU로 학습  |
| Data Parallel (데이터 병렬) | 모델 복사 N개 → 각 GPU가 다른 미니배치 학습 | 
| Model Parallel (모델 병렬)| 모델 자체를 여러 GPU에 나눠서 저장 |
|Pipeline Parallel| 모델을 여러 파트로 나누고 파이프라인처럼 순차 실행 |
|Tensor Parallel| 한 레이어 내부 연산을 여러 GPU가 나눠 처리 |
| Multi-Node (분산 학습)| GPU 여러 개가 여러 서버(노드) 에 흩어져 있음 |
|FSDP / ZeRO (DeepSpeed)| 모델, 그래디언트, 옵티마이저를 GPU 간 분할 저장  |
|Hybrid Parallel| 위 병렬 방식을 혼합  |






## 1. Multi-Node, DDP(Distributed Data Parallel) 
여러 대의 서버(노드)에 흩어진 GPU를 하나의 학습 클러스터처럼 연결하는 방식으로 세팅만 완료하면 한컴퓨터에 GPU 여러장 달린 것 처럼 쓸 수 있다.  
추가적으로 1번 서버에는 RTX 4090, 2번 서버에는 H100, 3번 서버에는 A100 이렇게 해두고, `batch size` 를 VRAM 가장 작은거 기준으로 세팅해서 사용도 가능하다고 한다(권장 방식은 아님).  

### 1.1 세팅 방법

환경 세팅
* 모든 서버: 동일 코드 + 동일 데이터셋 경로(또는 동일 스토리지 마운트).
* CUDA/드라이버, PyTorch, Accelerate, Pytorch 등등 버전 통일.

네트워크 준비
* 서버 간 통신 가능 IP 확인 + 포트 29500(또는 지정 포트) 열기

accelerate 세팅은 `accelerate config` 에서 하든, 코드 내부에서 `accelerate launch` 플래그로 하든 둘 중 하나만 쓰던가 똑같은값 적용해줘야함.  

총 3대의 컴퓨터가 있고, 이미 데이터, 코드, python 환경 다 동일하게 세팅했다고 가정
* 1번 서버 IP: 1.1.1.1
* 2번 서버 IP: 2.2.2.2
* 3번 서버 IP: 3.3.3.3

그리고 각 서버마다 수동으로 학습 코드를 실행 시켜 주면 3개의 프로세스가 렌데부 성공하는 순간 자동으로 학습을 시작한다.

**accelerate 세팅**
1. `accelerate config` 기반
```bash
- In which compute environment are you running?: Multi-GPU
- Which type of machine are you using?: 3
- Machine rank? → (각 서버에서) 0 / 1 / 2
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16 
- What is the main process IP? : 1.1.1.1
- What is the main process port? : 29500
accelerate launch train.py ...
```


2. `accelerate 플래그` 기반
```bash
# 1번 컴퓨터
accelerate launch \
  --multi_gpu --num_machines 3 --machine_rank 0 \
  --main_process_ip 1.1.1.1 --main_process_port 29500 \
  --mixed_precision bf16 \
  train.py --your_args ... 
# 2번 컴퓨터
accelerate launch \
  --multi_gpu --num_machines 3 --machine_rank 1 \
  --main_process_ip 1.1.1.1 --main_process_port 29500 \
  --mixed_precision bf16 \
  train.py --your_args ... 
# 3번 컴퓨터
accelerate launch \
  --multi_gpu --num_machines 3 --machine_rank 2 \
  --main_process_ip 1.1.1.1 --main_process_port 29500 \
  --mixed_precision bf16 \
  train.py --your_args ... 
```

`accelerate config` 기반 or `accelerate 플래그` 기반 둘 중 하나만 하면 됨.  
추가로 각 서버별로 통신하는 시간인 DDP timeout을 다르게 가져가고 싶으면 코드 내부에서 accelerate 설정 바꿔주면됨
```python
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=60))]
accelerator = Accelerator(kwargs_handlers=handlers)
```

근데 테스트 해볼 환경이 있어야 해보는데, 직접해보고 수정은 다음 기회에...  

그리고 실행시 반드시 `accelerator.prepare`로 감싸주어야하는데, 이제 이게 뭐하는 건지 이해가됨.  
그리고 log나 wandb로 보낼 때는 `accelerator.is_main_process` 로 걸러내서 진행 하는데 이것도 로그 출력시킬 때 main만 하도록 해야할듯. 
* 전체 속도는 가장 느린 GPU에 맞춰진다.

```python
from accelerate import Accelerator

accelerator = Accelerator()  # mixed_precision, dynamo 등은 config/플래그로
model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl) # 이걸 해줘야 세팅한 multi-node, DDP 가능

for step, batch in enumerate(train_dl):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs, batch["labels"])
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    if accelerator.is_main_process and step % 100 == 0:
        accelerator.log({"train/loss": loss.item()}, step=step)

# 저장
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), "model.pt")
```

## 2. FSDP / ZeRO (DeepSpeed)
FSDP와 ZeRO는 DDP를 개선해서 “메모리를 아끼는” 분산 학습 기법임. DDP는 모델 전체 복사본을 GPU에 올리는데 모델이 크면 VRAM이 터진다.  
그래서 **샤딩(Sharding)**이 나왔는데, GPU끼리 모델 파라미터, 그래디언트, 옵티마이저 상태를 나눠서 저장함.  
GPU마다 모델의 일부분만 가지고 있기 때문에 VRAM 사용량이 1/N으로 감소되고, 필요한 것들만 통신으로 주고받음.
이것을 목적으로 만들어진게 `FSDP`와 `DeepSpeed` 인데 둘 중 하나를 `Distributed type` 에서 설정해주면 된다.

### 2.1 FSDP (Fully Sharded Data Parallel)
모델, 그래디언트, 옵티마이저를 전부 샤딩 후 forward/backward 중 필요한 shard만 GPU로 로드.  
메모리 절약 극대화, DDP 대비 VRAM 최대 80% 절감됨, 그러나 통신량이 많아 느려질 수 있음.

```bash
# accelerate config 에서 설정
accelerate config
# → Distributed type: FSDP
# → 나머지는 DDP랑 동일하게 설정 후 accelerate.prepare로 감싸준다.
```

### 2.2 ZeRO (DeepSpeed)

Microsoft가 만든 DeepSpeed의 핵심 기술로 FSDP와 원리는 같지만 더 세밀하게 나뉨.
* Stage 1: 옵티마이저 상태만 샤딩
* Stage 2: 옵티마이저 + 그래디언트 샤딩
* Stage 3: 모델 파라미터까지 샤딩 (FSDP와 거의 동일)
* Stage 4: 일부 상태를 CPU나 NVMe로 옮겨 VRAM 추가 절약

```bash
# accelerate config 에서 설정
accelerate config
# → Distributed type: DEEPSPEED
# → DeepSpeed config file: ./ds_config.json
# → 나머지는 DDP랑 동일하게 설정 후 accelerate.prepare로 감싸준다.
#ds_config.json 예시
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu" }
  },
  "train_batch_size": 64,
  "gradient_accumulation_steps": 4,
  "bf16": { "enabled": true }
}
```
요약하면  
* FSDP = “PyTorch 기본 샤딩 DDP.”
* ZeRO (DeepSpeed) = “샤딩을 더 세밀히 제어하고 CPU/NVMe까지 활용하는 확장판.”

둘 다 목적은 같다 — VRAM 절약 + 대형 모델 학습 가능하게 하지만 속도는 느려진다. 아래는 ChatGPT가 실제 체감 예시써줬는데 상당히 괜찮은거같다.  

| 구성 | GPU VRAM 합계 | 모델 크기 | DDP 가능 배치 크기 | FSDP / ZeRO 가능 배치 크기 |
|------|----------------|------------|--------------------|-----------------------------|
| 4 × A100 (80GB) | 320GB | 10B 파라미터 | 1 ~ 2 | 16 ~ 32 |
| 4 × RTX 4090 (24GB) | 96GB | 2B 파라미터 | 2 ~ 4 | 16 이상 |


아래처럼 Diffusion 학습할 때 로스 등락이 거의 개잡주 횡보장처럼 나올 때가 많은데 (timestep별로 loss차이가 큼), 이 때 batch가 좀 컸으면 학습이 더 안정될텐데 라고 생각한 적이 많았음....  
Multi-Node 환경이 가능할 때 한번 해봐야겠다.

![FLUX-Dev Omini-Control Loss](https://private-user-images.githubusercontent.com/26128046/512207253-8f036790-4395-4c3e-a556-5b7100735026.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjI3ODUxNDYsIm5iZiI6MTc2Mjc4NDg0NiwicGF0aCI6Ii8yNjEyODA0Ni81MTIyMDcyNTMtOGYwMzY3OTAtNDM5NS00YzNlLWE1NTYtNWI3MTAwNzM1MDI2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEwVDE0MjcyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVhMmUwNDkyNTYxZjc3OTgzOWM2OTU4Mjk5MjNhMWE4OTg4OGY2NDc3NTcxMjBhZmFjOWZkNDk2OGMyOWMwY2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.D3mihbKJuBMB1XAqoZG7yBq_JNhIs1Fasp5VLGsAkdY)

## 3. Model Parallel & Pipeline Parallel
모델을 여러 GPU에 분할해 한 번의 forward를 나눠 처리가 목적(VRAM 절약). 속도는 GPU 간 텐서 이동 때문에 느려진다 (물론 안써봄).  
이 개념을 잘못 알고 있던게, 2개의 GPU를 하나의 GPU처럼 쓴다로 오해하고 있었음.  

모델 자체가 하나의 VRAM안에 안들어갈 때 쓰는데, 사실 요즘 이미지 모델은 커봐야 H100선에서 처리가 가능하지만 RTX 같은곳에서는 돌리기 무리가 있기에 알긴 해야할 듯한다. 물론 RTX로 모델 학습을 하는거 자체가 말이 안되긴하는데 추론에도 사용할 수는 있을테니까 알긴해야함ㅇㅇ.

| 구분 | Model Parallel | Pipeline Parallel |
|------|----------------|------------------|
| **기본 개념** | 모델을 여러 GPU에 나누어 저장 (레이어 단위 분할) | 모델을 나누되, 여러 **micro-batch**를 겹쳐 동시에 흐르게 함 |
| **실행 순서** | GPU0이 계산 후 결과를 GPU1로 전달 → 순차 실행 | GPU0은 1번째 micro-batch, GPU1은 0번째 micro-batch를 동시에 처리 |
| **GPU 활용률** | 낮음 (대부분 GPU가 대기) | 높음 (동시 실행으로 효율적) |
| **통신량** | 적음 | 많음 (파이프라인 스케줄링 필요) |
| **속도** | 느림 | 빠름 |
| **대표 구현체** | 수동 `.to("cuda:x")` 분할, 기본 PyTorch 방식 | DeepSpeed, Megatron, FairScale 등의 파이프라인 엔진 |
| **주 사용 목적** | 모델이 한 GPU에 안 들어갈 때 단순 분할 | 대형 모델 학습 시 GPU 활용률을 높이고 속도 개선 |

근데 Model Parallel은 구형 방식인듯함, 많이 안쓰일듯. Hybrid 로 DeepSpeed + Pipeline Parallel을 쓰면 
* 0번 GPU: 모델 토막 1
* 1번 GPU: 모델 토막 2
* 3번 GPU: 모델 토막 나머지
* 4번 GPU: 그래디언트
* 5번 GPU: 옵티마이저
뭐 대략 이런식으로 토막내서 뿌린다고 한다.  

### 3.1 Model Parallel
```python
import torch.nn as nn
import torch

class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Linear(4096, 4096).to("cuda:0")
        self.part2 = nn.Linear(4096, 4096).to("cuda:1")

    def forward(self, x):
        x = self.part1(x)
        x = x.to("cuda:1")
        x = self.part2(x)
        return x

    
ds_config = {
    "zero_optimization": { "stage": 2 },
    "train_batch_size": 16
}
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")

x = torch.randn(32, 4096, device="cuda:0", pin_memory=False)
y = model_engine(x)                                      # GPU0
loss = y.float().pow(2).mean()
loss.backward()                                # 자동으로 역전파 시 디바이스 간 통신 포함
optim.step()
optim.zero_grad()
```
이건 뭐 솔직히 쓸일이 없을듯하다. 그리고 보면 0번 GPU가 일하는 동안 1번 쉬고, 1번이 일하면 0번이 쉬게되는 상황이 발생하는데, 이를 개선하기 위해 나온것이 Pipeline Parallel이라고 한다. 
### 3.2 Pipeline Parallel
모델을 GPU별로 분할하는 건 동일하지만, 배치를 micro-batch 단위로 잘라서 여러 GPU가 동시에 일하게 함. 그리고 DeepSpeed가 자동 스케줄링을 관리한다.  

```python
import torch.nn as nn
from deepspeed.pipe import PipelineModule, LayerSpec

class Block(nn.Module):
    def __init__(self, d=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU())
    def forward(self, x): return self.net(x)

def build_pipeline_model(num_layers=12, num_stages=4):
    layers = [LayerSpec(Block) for _ in range(num_layers)]
    # num_stages = 파이프라인 스테이지 수 = 사용할 GPU 수
    model = PipelineModule(
        layers=layers,
        loss_fn=nn.MSELoss(),     # 라벨이 있으면 커스텀 loss_fn 가능
        num_stages=num_stages,
        partition_method="parameters",  # 파라미터 균등 분할
        activation_checkpoint_interval=0
    )
    return model

accelerator = Accelerator()  # DeepSpeed/분산/정밀도는 launch가 주입
num_stages = accelerator.num_processes  # GPU 수 = 스테이지 수(단일-파이프라인 전제)

model = build_pipeline_model(num_layers=12, num_stages=num_stages)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dl = get_data()

# DeepSpeed 파이프라인은 model을 DeepSpeed 엔진으로 감싼다(Accelerate가 내부 처리).
model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

for step, batch in enumerate(dl):
    x, y = batch
    # PipelineModule의 loss_fn이 있으면 (x, y) 튜플을 그대로 넘겨도 된다.
    loss = model(x, y)
    accelerator.backward(loss)
    optimizer.step(); optimizer.zero_grad()

    if accelerator.is_main_process and step % 100 == 0:
        accelerator.print(f"step {step} loss {loss.item():.4f}")
```
이렇게 학습 파이프라인을 만들어둔 뒤 deepspeed용 json을 만들어 준다.
```json
// ds_config.json
{
  "bf16": { "enabled": true },
  "train_micro_batch_size_per_gpu": 8,      // 마이크로배치 크기(스테이지당)
  "gradient_accumulation_steps": 4,         // 누적스텝
  "zero_optimization": { "stage": 2 },      // ZeRO 병행 가능(2 또는 3)
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}

```
이 때 전역 배치 크기 = train_micro_batch_size_per_gpu × gradient_accumulation_steps × 데이터병렬도(world_size / num_stages)
파이프라인만 쓰면 데이터병렬도=1.  

```bash
# 서버0
accelerate launch \
  --multi_gpu --num_machines 2 --machine_rank 0 \
  --main_process_ip 1.1.1.1 --main_process_port 29500 \
  --deepspeed_config_file ./ds_config.json \
  train.py

# 서버1
accelerate launch \
  --multi_gpu --num_machines 2 --machine_rank 1 \
  --main_process_ip 1.1.1.1 --main_process_port 29500 \
  --deepspeed_config_file ./ds_config.json \
  train.py
```
`ds_config.json` 만 만들어주고 모델만 감싸주면 딱히 뭐 할껀 없는듯... 진짜 복잡해보이는거 코드 몇줄로 뚝딱 해주는 accelerate 대단하긴하다.
## 5. Tensor Parallel
한 레이어 내부 연산을 여러 GPU가 나눠 처리할 정도로 엄청 큰 규모의 모델에서 쓰인다. 솔직히 이건 쓸일 없을듯 해서 넘어간다. 

마지막 accelerate 포스트인, 각종 accelerate 함수 unwarp, prepare 등등을 봐야겠다.

[3편]({% post_url 2025-11-10-Accelerater %})에 계속
 