---
layout: post
title:  "Accelerater 테이크 다운"
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






# 1. 설치 및 세팅
처음 `accelerater` 설치 후 config 세팅

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

