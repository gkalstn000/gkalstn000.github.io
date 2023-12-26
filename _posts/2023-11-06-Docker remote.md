---
layout: post
title:  "Remote 서버에 Docker 활용하여 작업하기"
categories: 기타
date: 2023-11-06 11:40:18 +0900
tags: Docker
mathjax: True
author: Haribo
---
* content
{:toc}
















## 0. Docker 설치

1. [Docker 설치](https://docs.docker.com/engine/install/ubuntu/)
2. [Nvidia-Docker 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. [Docker storage 옮기기](https://evodify.com/change-docker-storage-location/) (default : `/var/lib` ) 
   * Docker image가 상당히 많은 용량을 잡아 먹는데 `/var` 폴더가 Docker 이미지가 저장되는 default 위치. 
   * `/var` 의 용량 부족으로 image 다운로드 불가능한 경우가 많아 여유있는 공간을 Docker 이미지가 저장되는 위치로 변경해줘야함.

## 1. Image 만들기

```bash
git clone <https://github.com/cnstark/pytorch-docker.git>
cd pytorch-docker

# python generate_build_script.py --os <ubuntu or centos> --os-version <e.g. 20.04, 8> --python <e.g. 3.9.12> --pytorch <e.g. 1.9.1> --cuda <e.g. 11.1, cpu>
# Docker Image script
python generate_build_script.py --os ubuntu --os-version 20.04 --python 3.9.12 --pytorch 1.13.0 --cuda 11.7
# Docker Image Build
sh scripts/build_1.7.1_py3.8.13_cuda10.2_ubuntu18.04.sh
```

## 2. Container 설정 (Remote 환경에서)

### Container 실행

PyTorch Image:Tag

- cnstark/pytorch:1.7.1-py3.8.13-cuda10.2-ubuntu18.04

mount 방법

- -mount type=bind,source=(서버 폴더 절대경로),target=(Container에서의 작업 경로)

port 노출 방법

- p (host의 port):(container의 port)
- container는 하나의 가상환경이며 host machine과 container의 port를 연결해줘야 합니다. 예를 들어 host machine에서는 8089 port를 열고 container의 7001 port를 연결해주기 위해서는 -p 8089:8089 명령어가 필요합니다

GPU 연결 방법

- -gpu "device=0" 또는 --gpus all
- 연결하고자 하는 device id는 nvidia-smi로 확인가능(맨 왼쪽 0 또는 1로 표시됨)

```bash
#docker run -d -it --name {Container_Name} --gpus all -p {Host_Pnum}:{Container_Pnum} --mount type=bind,source={Host_workspace},target={Container_workspace} {Image_Name}:{Tag}

docker run -d -it --name nted --gpus all --privileged -p 8088:8088 --mount type=bind,source={SOURCE_PATH},target=/root

docker start CONTAINER_NAME
docker attach CONTAINER_NAME
```

### Container bash 필요 라이브러리 설치

```bash
apt-get update  
apt-get install nano net-tools openssh-server curl gcc g++ vim git
nano /etc/ssh/sshd_config
```

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/283192500-b6f61251-07a2-4581-bc14-bc4a4da3a0dc.png">     
    <figcaption>Pychar local-remote(docker) 예시 </figcaption>   
  </figure> 
</div>

* Port: 설정한 port 번호로 수정
* `PermitRootLogin yes` 로 수정

 부분 수정 후 **Ctrl+X를 누르고 Y를 눌러** 내용을 저장

### Password 설정

```bash
passwd root 
```

### SSH service 실행 후 테스트

```bash
service ssh start
```



본인 Local 노트북 터미널에서 server에 실행되고 있는 container에 접속

* 외부에서 컨테이너로 바로 접속

```bash
ssh root@<서버 IP> -p <Container Port>
```

### Anaconda 설치

```bash
curl <https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh> --output anaconda.sh
sh anaconda.sh
source ~/.bashrc
```

### Container 종료

`Ctrl+p` 후 `Ctrl+q` 로 종료.

**매우중요. Container background 상태 유지시키며 bash 접속만 종료해야함.**

- Docker 꺼졌을 때 컨테이너의 ssh 재부팅 해줘야함

  ```bash
  docker star [container_ID or container_name]
  docker attach [container_ID or container_name]
  
  # container 접속 후
  service ssh restart
  ```

## 3. IDE SSH 와 Container 연결

Server SSH 연결하듯 동일하게 진행

- Host: Server IP
- Port: Container port_num
- IP: root

<div style="text-align: center;">   
  <figure>     
    <img src="https://user-images.githubusercontent.com/26128046/283191961-700c03f4-822c-45a9-91b5-39e4a9b34559.png">     
    <figcaption>Pychar local-remote(docker) 예시 </-></figcaption>   
  </figure> 
</div>

## 4. 그 외 디테일한 설정

- pytorch cuda는 available 하지만 /usr/local/ 에 cuda 가 있어야하는데 없음. 필요하다면 재설치
- CUDA 환경변수
