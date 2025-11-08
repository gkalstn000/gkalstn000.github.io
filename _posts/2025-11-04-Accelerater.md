---
layout: post
title:  "Accelerater 테이크 다운 (1편, 세팅 및 학습 가속)"
categories: 실습
date: 2025-11-04 11:40:18 +0900
tags: AI Optimizer 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

내가 정말 리스펙 하면서 쥐어 패고싶은 [koyha-ss](https://github.com/kohya-ss) 코드를 공부하며 알게된 하드웨어 가속기 Accelerator.
주로 쓰는건 autocast, multi-gpu 이정도만 알고 나머지는 쟤가 써놓은대로만 써왔음.  

처음 세팅할 때 `accelerater config` -> 코드 내부에 다양한 accelerater 함수들 (unwrap, log, prepare 등등) -> 그리고 python 실행할 때 쓰는 accelerater 옵션들 (--mix-precision 같은거).  
`kohya-ss` 이 사람 코드 기준으로 하나하나 알아보려고 한다.  






# 0. 시작
Diffusion 학습을 위한 `kohya-ss` 코드 ([sd-scripts](https://github.com/kohya-ss/sd-scripts), [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)) 를 보면, 처음 `accelerater` 설치 후 config 세팅을 해준다.

```bash
accelerate config

- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

이렇게 터미널에서 config 만진 다음에 학습 코드 내부에서 다시 한번 아래와 같이 한번더 accelerater를 만진다.

```python
from accelerate.utils import TorchDynamoPlugin, set_seed, DynamoBackend
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)
    return accelerator
```

그리고 학습을 실행할 때 다시 아래와 같이 mixed_precision, thread 같은거를 다시한번 만져준다.
```bash
accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 train.py ~~~
```

우선 세팅부터 어느것이 제일 기준이고, 어느것이 덮어씌우는 건지 파악을 해야한다.  
게다가 `wandb`도 accelerater 안에 집어넣고 `accelerater.log` 로 `wandb`로 쏴버린다.  
그리고 굳이 `logger` 나 `print` 안쓰고 `accelerater.print` 로 출력시킬 때도 있음.

---

# 1. accelerate 세팅 우선 순위
코드를 보면 `accelerate config`, 터미널 명령어, 코드 내부 `def prepare_accelerator(...)` 에서 precision은 뭐고, DDP는 어떻고, device는 뭐고, torch 가속은 어떻게 할 것이며 중복적으로 설정 하고 있다.  
  
순번이 높을수록 우선순위 높은거 (덮어씌우는 놈)
1. 코드 내부 `Accelerator(...)`로 설정
2. `accelerate launch` 플래그
3. `accelerate config` 설정

중요한것은 config에서는 `float32`로 했다가, 코드안에서 `bfloat16` 이런식으로 하는 것은 권장되지 않는다.  

요약:
* “무조건 이겨야 할 값”은 코드의 Accelerator(...).
* 비워두면 그다음은 launch 플래그, 그다음이 config.

---

# 2. accelerate 세팅(플러그) 값 살펴보기
아직 대규모 foundation 모델, 억단위의 데이터파이프라인 처리를 해본적 없어서 분산학습 관련 기능들을 제대로 공부한적이 없다.  
그리고 `Torch Dynamo`, `DeepSpeed`, `numa efficiency` 등 얘네는 뭔지도 모르겠어서 알아보기로함.
* 지금까지는 foundation 모델에 LoRA, Omini-Control 같은거 붙여서 십만장 단위 데이터셋만 학습시켜서 굳이 알아보지 않았었음.

우선 accelerate는 크게 분산학습, 학습가속으로 나눠볼 수 있음.
분산 학습: `DeepSpeed`, `InitProcessGroupKwargs`, `DistributedDataParallelKwargs`, `PartialState`
학습 가속: `Torch Dynamo`, `numa efficiency`

## 학습 가속

### 2.1 Torch Dynamo
이거 뭔지모르고 좋아보여서 써보려고 `accelerate config`에서 아래와 같이 쓸꺼냐 라고 물어서 `yes` 했었는데 그래프 어쩌고 저쩌고 해서 겁먹고 걍 안썼던 기억이있다.  

```bash
Do you wish to optimize your script with torch dynamo? [yes/NO]: yes
Which backend do you want to use? [inductor]: 
Which mode do you want to use? [default]:
Do you want full graph optimization? [no]:
Do you want dynamic shape support? [no]:
```

`Torch Dynamo`는 **“한 번에 묶어서 실행”** 이게 핵심인듯하다. 

```python
for step, (x, y) in enumerate(loader):
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
```
이런 코드가 있을 때 python은 한줄읽고 GPU 로 보내고 GPU가 처리하고 다시 돌려주고 이걸 한줄한줄 반복하는데, 여기서 GPU는 개빨라서 이미 처리하고 다음 명령 기다리는데 python 은 개느려서, 보내고 받고에서 GPU가 놀아버리는 병목 현상이 발생한다고한다.  
Python 코드를 실행하기 직전에, Torch Dynamo가 끼어들어 PyTorch 연산인 부분들을 묶어서 하나의 덩어리(그래프)로 만들고 이제 한줄씩이 아니라 통째로 GPU로 보내고 받고 한다. 
> Python 코드  →  그래프(FX IR)  →  컴파일(Inductor 등)  →  GPU용 실행코드  
> Python이 한 줄씩 지시하던 걸 Torch Dynamo가 대신 통째로 계획 세워서 GPU에 맡긴다.

세팅 방법은 `accelerate config`에서도 가능하긴한데, 어차피 코드내부에서 accelerate 세팅이 더 파워 쌔니까 그거만 확인해봤음.
```python
from accelerate.utils import TorchDynamoPlugin, DynamoBackend
from accelerate import Accelerator

dynamo_plugin = TorchDynamoPlugin(
    backend=DynamoBackend("INDUCTOR"), # 컴파일러, nvfuser, onnxrt 등은 실험용
    mode="default", # 그래프를 얼마나 공격적으로 최적화할지, reduce-overhead / max-autotune 등은 실험용.
    fullgraph=False, # Dynamo가 부분 그래프만 잡음, full-graph 잡으면 실패 가능성 높음
    dynamic=False, # True면 입력 shape가 자주 변할 때 자동 대응 (batch 1일 때 하는건가봄)
)
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
for batch in dataloader:
    with accelerator.accumulate(model):
        out = model(batch)
        loss = cal_loss(out, batch.get('gt'))
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

Diffusion 테크리포트 보면 foundation 모델 초반 학습에는 `256x256` -> `512x512` -> `768x768` 하다가 마지막에 고품질 **가변 해상도** 학습을 하는데, 가변 해상도라해도 버켓으로 묶어서 배치단위로 처리해주면 쓸 수 있다고 한다. 

당연히 단점도 존재한다.
* 초기 지연: 첫 컴파일 때 몇 초 느림
* VRAM 증가: 그래프 캐시로 GPU 메모리 추가 사용
* 동적 구조 제약: 입력 shape나 분기문이 자주 바뀌면 그래프 재컴파일, 속도↓
* 디버깅 어려움: 컴파일된 그래프 내부는 step 단위 추적 불가

### 2.2 DeepSpeed
CPU가 여러 개(2개 이상) 달린 서버에서 각 GPU가 가까운 CPU 메모리만 쓰게 해서 데이터 로딩 병목을 줄이는 가속 옵션

얘는 딱히 단점이 없어서 멀티-CPU면 걍 키는게 정배인듯함.



| 항목  | 장점 | 단점 | 권장 상황 |
|-------|---------------------------|------|------|------------|
| **Torch Dynamo**  | 속도 향상, 코드 수정 불필요 | 첫 실행 지연, VRAM 추가 사용, 동적 구조에 약함 | 단일 GPU·고정 해상도 학습 |
| **NUMA Efficiency** | CPU–GPU 데이터 이동 지연 감소, 데이터 로딩 효율 향상 | Multi-CPU 리눅스 전용 | 다중 CPU 소켓 서버(워크스테이션·GPU 서버) |

[2편]({% post_url 2025-11-08-Accelerater %})에 계속
