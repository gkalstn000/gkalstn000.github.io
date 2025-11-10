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
    dynamo_backend = args.dynamo_backend
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


