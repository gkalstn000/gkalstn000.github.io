---
layout: post
title:  "Accelerater 참수 (3편, 각종 함수들)"
categories: 실습
date: 2025-11-09 11:40:18 +0900
tags: AI Optimizer 학습 training
mathjax: True
author: Haribo
---
* content
{:toc}

[koyha-ss](https://github.com/kohya-ss) 가 쓴 다양한 accelerate 실전 코드를 살펴본다.

# 1. accelerate 세팅
`acclerate config`나 `accelerate 플래그`로 쓰기 쫌 복잡한애들을 코드로 세팅한다.  
공부하고 나니까 코드가 읽힌다. 






```python
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState, DeepSpeedPlugin

def prepare_accelerator(args: argparse.Namespace, log_with=None, logging_dir=None):
    """
    this function also prepares deepspeed plugin
    """
    if args.seed is None or args.seed == -1:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    dynamo_backend = "NO"
    if args.torch_compile:
        dynamo_backend = args.dynamo_backend

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
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
    deepspeed_plugin = prepare_deepspeed_plugin(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_backend=dynamo_backend,
        deepspeed_plugin=deepspeed_plugin,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)
    if args.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0  # preventing overflow.
    if args.full_fp16 or args.fp16_master_weights_and_gradients:
        if args.offload_optimizer_device == "cpu" and args.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin
```

그러나 좀 길기도 하고, 처음보는 인자들도 있고 하니 좀 나눠서 봐야한다.

### 1.1 kwargs_handlers 및 기타 인자들
```python

init_process_group_kwargs = InitProcessGroupKwargs(
    backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
    init_method=(
        "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None),
    timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
)

ddp_kwargs = DistributedDataParallelKwargs(
    gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, 
    static_graph=args.ddp_static_graph
)

kwargs_handlers = [
    (init_process_group_kwargs if torch.cuda.device_count() > 1 else None),
    (ddp_kwargs if args.ddp_gradient_as_bucket_view or args.ddp_static_graph else None),
]
kwargs_handlers = [i for i in kwargs_handlers if i is not None]

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=log_with, # ''
    project_dir=logging_dir,
    kwargs_handlers=kwargs_handlers,
)
```

여기서 보이는 두놈 `InitProcessGroupKwargs`, `DistributedDataParallelKwargs` 에 대한 정보.
* `InitProcessGroupKwargs` = “DDP 통신을 시작할 준비” 세팅.
* `DistributedDataParallelKwargs` = “DDP로 돌릴 때의 동작” 세팅.
* DDP를 켜는 건 accelerate config/launch + Accelerator().prepare(...).

각각 인자부터 보면  

---
**InitProcessGroupKwargs**  
* backend: nccl(CUDA/NVIDIA), gloo(CPU/윈도우) 등
* init_method: env://(기본) 등 랜데부 URL
* timeout: 모든 rank가 모일 때까지 기다리는 시간

여기서 `init_method` 이놈이 쫌 꼬롬한 놈인데, 멀티 GPU 학습은 GPU마다 프로세스가 따로 실행되고 서로 통신하려면 서로 만날 장소(URL)가 필요한데 그 약속주소가 `init_method`이다.  

| 항목      | 의미               | 언제 씀             |
|---------|------------------|------------------|
| env://  | 환경변수 기반 초기화 (기본) | 이게 정배임           |
| file:// | 파일을 통해 초기화       | 공유 storage가 있을 때 |
| tcp://  | IP/Port 직접 지정    | 수동 설정 시          |

근데 accelerate launch 가 자동으로 환경변수 세팅하니까 `env://` 쓰면 된다.  

---
**DistributedDataParallelKwargs**  
* gradient_as_bucket_view: DDP는 여러 GPU가 gradient(기울기) 를 서로 주고받을 때, 버킷(bucket) 이라는 큰 임시 공간에 모아두는데 `True`를 주면 gradient 복사본을 안만들고 원본 메모리를 참조한다 ➡️ 즉, 메모리를 덜 쓴다.
* static_graph: 모델 구조나 입력 크기가 항상 똑같을 때, GPU 통신을 한 번만 계산해두고 재사용하게 해준다. 하지만 step마다 shape이 바뀌면(예: 가변 해상도 이미지) 이 최적화는 못 쓴다 — 오히려 오류나 경고 뜬다.

| 옵션                      | 하는 일             | 장점       | 주의점 |
|-------------------------|------------------|----------| ---|
| gradient_as_bucket_view | gradient 복사 안 만들고 원본 공유 | VRAM 절약  |일부 옵티마이저랑 충돌 가능|
| static_graph            | 모델 구조 고정으로 통신 캐싱 | DDP 속도 약간 향상 |입력 크기나 구조가 바뀌면 비활성화해야 함|


메모리 빠듯 → `gradient_as_bucket_view=True`   
완전 정적 모델(입력 shape/분기 불변) → `static_graph=True`  

---
**log_with**  
그냥 string 으로 tensorboard, wandb, trackio, comet_ml, aim, mlflow, clearml, dvclive, all, None 넣어주면된다. 다만 넣고나서 각 모니터링 라이브러리 초기화는 해줘야함.
```python

def init_trackers(accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str | None = 'Debug'):
    """
    Initialize experiment trackers with tracker specific behaviors
    """
    if accelerator.is_main_process:
        kst = ZoneInfo("Asia/Seoul")
        now_kst = datetime.now(kst)
        ymd_str = now_kst.strftime("%y%m%d")
        init_kwargs = {"wandb": {
            "entity": "msha",
            "name": f'{ymd_str}-{args.output_name}'
        }}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name if args.log_tracker_name is None else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

            # Define specific metrics to handle validation and epochs "steps"
            wandb_tracker.define_metric("epoch", hidden=True)
            wandb_tracker.define_metric("val_step", hidden=True)

accelerator = Accelerator(log_with="wandb")
init_trackers(accelerator, args)
```

### 1.2 dynamo_backend & deepspeed_plugin

torch dynamo 는 [Accelerater 1편]({% post_url 2025-11-04-Accelerater %}), DeepSpeed는 [Accelerater 2편]({% post_url 2025-11-10-Accelerater %}) 에서 보긴했는데 코드내부에서는 좀더 디테일한 설정이 가능한가보다.  
```python
def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)
    if args.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0  # preventing overflow.
    if args.full_fp16 or args.fp16_master_weights_and_gradients:
        if args.offload_optimizer_device == "cpu" and args.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin


dynamo_backend = "NO"
if args.torch_compile:
    dynamo_backend = args.dynamo_backend # "inductor" if args.torch_compile else "NO"
deepspeed_plugin = prepare_deepspeed_plugin(args)
accelerator = Accelerator(
    ...
    dynamo_backend=dynamo_backend,
    deepspeed_plugin=deepspeed_plugin,
)
```

헷갈리긴 하는데 [Accelerater 1편]({% post_url 2025-11-04-Accelerater %}) 에서 dynamo_plugin 은 그냥 코드 묶어서 처리하는 용도였는데 갑자기 dynamo_backend는 뭔가 싶다.  
* dynamo_backend="inductor" ← Accelerate가 내부에서 torch.compile() 실행
* 세부 제어: TorchDynamoPlugin(...) ← 모드·fullgraph·dynamic까지 직접 지정

```python
# Accelerater 1편 Plugin 방법 
dynamo_plugin = TorchDynamoPlugin(
    backend=DynamoBackend("INDUCTOR"), # 컴파일러, nvfuser, onnxrt 등은 실험용
    mode="default", # 그래프를 얼마나 공격적으로 최적화할지, reduce-overhead / max-autotune 등은 실험용.
    fullgraph=False, # Dynamo가 부분 그래프만 잡음, full-graph 잡으면 실패 가능성 높음
    dynamic=False, # True면 입력 shape가 자주 변할 때 자동 대응 (batch 1일 때 하는건가봄)
)
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
```


쉽게말해서 [Accelerater 1편]({% post_url 2025-11-04-Accelerater %})의 방법은 TorchDynamoPlugin(...)으로 백엔드 + mode + fullgraph + dynamic 등 세부 옵션까지 지정하는 디테일 방식이다.  
이 포스트의 방식인 backend는 dynamo_backend="inductor" 처럼 백엔드만 지정 → 내부에서 torch.compile(..., backend="inductor") 호출하는 간단 방식이다.  
즉, 둘 중 하나만 세팅하면 되는데 간단방식 or 디테일한 방식 정도로 알면 될듯하다.  

---

# 2. 학습 때 accelerate

## 2.1 Main process 구별
분산 학습을 할 때 그냥 logger, print, wandb로 토스를 해버리면 분산개수많큼 중복으로 처리를 해버린다.  

```python
is_main_process = accelerator.is_main_process
if is_main_process :
    something~~
```

그래서 보통 위와 같이 구별해야할 때 이런식으로 처리를 하는데 분산학습 관련 accelerate 기능들을 정리해봤다.

| API                                | 의미 | 언제 쓰나                                              |
|------------------------------------|------|----------------------------------------------------|
| accelerator.is_main_process        | 전 클러스터에서 rank==0 | 전역 로그, 체크포인트 저장, W&B 초기화 등 한 번만 해야 할 작업            |
| accelerator.print(...)             | 기본 전역 메인만 출력 | 중복 로그 방지                                           |
| accelerator.device                 | 이 프로세스가 쓸 torch.device | 메인 여부와 무관                                          |
| accelerator.is_local_main_process  | 각 노드별 로컬 rank==0이면 True | 노드 로컬 캐시 생성, 노드별 로그 파일 등 “노드마다 한 번만” 해야 할 작업에 사용.  |

아래는 내가 안써본 것들 (분산쪽)

| API | 의미 | 언제 쓰나 |
|-----|------|-----------|
| accelerator.is_local_main_process | 각 노드(머신)에서 local_rank==0 | 노드별 파일 캐시 생성, 노드 로컬 로그 남길 때 |
| accelerator.process_index | 전역 rank (0..world_size-1) | 프로세스별 분기 처리 |
| accelerator.local_process_index | 노드 내 로컬 rank (0..local_size-1) | 노드 내 분기 처리 |
| accelerator.num_processes | 전역 world size | 유효 배치 계산 등 |
| accelerator.local_num_processes | 노드당 프로세스 수 | 노드별 리소스 계산 |
| accelerator.main_process_first() | 컨텍스트 매니저. 메인이 먼저 실행, 나머지는 대기 | 데이터 다운로드, 디렉터리 생성 순서 보장 |
| accelerator.local_main_process_first() | 노드별 메인이 먼저 실행 | 멀티노드에서 노드별 준비 작업 |

`device` 바꿀 때도, 출력할 때도 그냥쓰면 안되고 accelerator가 주는대로 해줘야한다.

## 2.2 Model 관련
#### accelerator.unwrap_model(model) <- 진짜 많이 쓰임
* DDP/FSDP/DeepSpeed로 감싸진 래퍼를 벗겨 원본 nn.Module을 반환.
* 체크포인트 저장·가중치 직접 접근 전에 필수.

`unwrap` 하고나서 보통 sampling, 가중치 저장을 한다.


## 2.4 학습 관련
#### accelerator.accumulate
배치크게 뻥튀기 하는 용도. accumulate 스텝만큼 loss를 계산하여 한번에 업데이트.  
속도는 느려지지만 학습 안정화를 위해 필수
```python
# accumulate step 정의
Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    ...
)
...
for step, batch in enumerate(dl):
    with accelerator.accumulate(model):
        out = model(**batch)
        loss = out.loss
        accelerator.backward(loss)
        optimizer.step(); optimizer.zero_grad()
```

#### accelerator.sync_gradients
accelerator.accumulate 안에서만 쓸 수 있음. accumulate에서 마지막 gradient 동기화 끝나고 뭔가를 해야할 때 얘 걸어놓고 쓴다.  
```python
with accelerator.accumulate(model):
    ...
    if accelerator.sync_gradients:
        accelerator.log({"loss": loss.item()}, step=global_step)
```
#### accelerator.backward
그냥 loss.backward() 대신 쓰는 버전. 자동으로 mixed precision(fp16/bf16) 이나 분산학습 환경에 맞게 동작시켜줌.


#### accelerator.clip_grad_norm_
gradient norm 클리핑 유틸로 래핑된 모델에서도 바로 사용 가능. 보통 `max_norm = 1`, `max_norm = 0.5` 많이 씀.
```python
with accelerator.accumulate(model):
    ...
    accelerator.backward(loss)
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); optimizer.zero_grad()
```
#### accelerator.wait_for_everyone
모든 분산 프로세스가 여기까지 도달할 때까지 기다림. 모델 저장이나 평가 전에 씀.  
안쓰면 레이스·불일치·가끔 데드락 난다. 학습 자체는 돈다. “공유 작업” 앞뒤에서만 필수.
```python
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.save(...)

...
accelerator.wait_for_everyone()
optimizer.eval()

```
#### accelerator.autocast
쓰면 매우 편함. 굳이 데이터셋 type 안바꿔도됨.  
다만 device는 자동으로 안바꿔주니까 수동으로 해야함.
```python
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
...
)
with torch.set_grad_enabled(is_train), accelerator.autocast():
    out = model(**batch)
    loss = loss_fn(out, target)
accelerator.backward(loss)

# 혹은 아래처럼 샘플링
with accelerator.autocast(), torch.no_grad():
    sample = sampling(model, *batch)
```
## I/O 관련
#### accelerator.register_save_state_pre_hook & accelerator.register_load_state_pre_hook
만약 foundation 모델에 adapter (LoRA, ControlNet 등등) 붙여서 학습했을 때,  모델 저장/로드는 adapter 만 필요하므로 state 저장 전에 저장/로드 할 것들 전처리 해줄 때 쓰인다.  

```python
def save_model_hook(models, weights, output_dir):
    # pop weights of other models than network to save only network weights
    # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
    if accelerator.is_main_process or args.deepspeed:
        remove_indices = []
        for i, model in enumerate(models):
            if not isinstance(model, type(accelerator.unwrap_model(training_model))):
                remove_indices.append(i)
        for i in reversed(remove_indices):
            if len(weights) > i:
                weights.pop(i)
        # print(f"save model hook: {len(weights)} weights will be saved")

    # save current ecpoch and step
    train_state_file = os.path.join(output_dir, "train_state.json")
    # +1 is needed because the state is saved before current_step is set from global_step
    logger.info(
        f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value + 1}")
    with open(train_state_file, "w", encoding="utf-8") as f:
        json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

steps_from_state = None

def load_model_hook(models, input_dir):
    # remove models except network
    remove_indices = []
    for i, model in enumerate(models):
        if not isinstance(model, type(accelerator.unwrap_model(training_model))):
            remove_indices.append(i)
    for i in reversed(remove_indices):
        models.pop(i)
    # print(f"load model hook: {len(models)} models will be loaded")

    # load current epoch and step to
    nonlocal steps_from_state
    train_state_file = os.path.join(input_dir, "train_state.json")
    if os.path.exists(train_state_file):
        with open(train_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps_from_state = data["current_step"]
        logger.info(f"load train state from {train_state_file}: {data}")

accelerator.register_save_state_pre_hook(save_model_hook)
accelerator.register_load_state_pre_hook(load_model_hook)

...
# 저장할 때
accelerator.save_state(state_dir, args=args, accelerator=accelerator)
# 로드할 때
accelerator.load_state(state_dir)
```

#### accelerator.save_state
분산 안전 체크포인트 저장. 가중치 뿐만 아니라 옵티마이저, 스케줄러, AMP 스케일러, RNG 상태 등을 랭크0 기준으로(또는 샤딩 방식에 맞게) 디스크에 씀. DeepSpeed/FSDP도 맞춰 저장.
```python
save_dir = f"ckpts/step_{step}"
accelerator.save_state(save_dir)
accelerator.wait_for_everyone()  # 다른 랭크 동기화
```



## 2.3 log 관련

#### accelerator.trackers & accelerator.log
accelerator 에 넣어 둔 로깅 라이브러리 불러오거나, log 기록할 때 쓰임
```python
logs = {"avr_loss": avr_train_loss} 
accelerator.log(logs, step=global_step)
```
그냥 사전에 key-val 박아두면 알아서 해준다. 다만, 이미지 오디오 같은 특수한 결과값은 tracker 뽑아온 뒤 각 tracker에 맞게 후처리 해준 다음에 넣어줘야함.  
```python
wandb_tracker = accelerator.get_tracker("wandb")
wandb_tracker.log({f"sample_{enum}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption
```

공부하고 나서 보니 코드 개판인 부분이 있어서 고쳐놔야겠다.  
이것으로 accelerate 해체쇼를 마친다.