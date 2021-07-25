---
layout: post
title:  "jupyter notebook Docker를 이용한 서버 컴퓨터 원격 접속 "
categories: 기타
date:   2021-07-25 00:10:18 +0900
tags: 일상
mathjax: True
author: Haribo
---
* content
{:toc}


> Docker에 대한 설명은 생략하고 오로지 원격 접속 하는 방법에 대해서만 글을 쓸 예정임.
>
> SSH나 Docker에 대한 설명은 아래 참고사이트가서 보고 오시길 바람.
>
> 그리고 MAC 기준 설명임. 윈도우는 할줄모름

## 참고사이트

### SSH

https://eungbean.github.io/2019/03/22/jupyter-ssh/

http://programmingskills.net/archives/315

### Docker

[블로그1](https://greeksharifa.github.io/references/2021/06/21/Docker/)

[블로그2](https://89douner.tistory.com/96)

[구글 클라우드](https://cloud.google.com/ai-platform/training/docs/custom-containers-training?hl=ko)

https://towardsdatascience.com/using-jupyter-notebook-running-on-a-remote-docker-container-via-ssh-ea2c3ebb9055

# SSH 연결

터미널에서 

```
ssh-keygen
```

입력 후 엔터누르면됨. 중간에 `passphrase` 설정하라는게 나오는데 비밀번호 같은것으로 생각된다. 굳이 안해도 접속 가능함. 결과물 이미지로 보여주고 싶지만 뭔가 보여주면 안될것 같은 비주얼이라 생략. 그리고 ssh를 통해 서버에 접속한다.

```
ssh -L <로컬 포트번호>:localhost:<서버 포트번호> <server_name>@<server_IPaddr> -p <ssh_port_num>

# 로컬 포트번호, 서버 포트번호는 서버 이용자들끼리 안겹치게 설정
```

예를들면 이런식으로 접속

```
ssh -L 9999:localhost:151515 hello@1.2.3.4 -p 22
```



### 다른 포트번호로 SSH 연결

ssh는 기본 포트가 22번인데 22번말고 다른 포트 번호로 접속하고 싶으면

```
vi /etc/ssh/sshd_config
```

를 실행해서

```
#Port 22
Port 141414  <- 새로운 포트번호 입력하면됨
```

를 써주면 된다. 그리고

```
# 새로 등록한 포트는 추가
sudo ufw allow 151515/tcp

# 기존 SSH 포트는 차단
sudo ufw deny 22/tcp

# 서비스를 재시작 합니다.
service ssh restart
```

그런다음

```
ssh -L 9999:localhost:151515 hello@1.2.3.4 -p 141414
```



# Notebook 접속

GPU를 위한 `nvidia-docker`와 필요한 [이미지](https://hub.docker.com/r/pytorch/pytorch) pull은 완료했다는 가정하에 설명을 진행한다. 안해쓰면 [여기](https://greeksharifa.github.io/references/2021/06/21/Docker/)가서 `nvidia-docker`와 원하는 이미지를 다운 받고 보면 된다. 우선 현재 대략적인 상황을 이미지로 표현하면

![image-20210725145217674](/images/Docker/image-20210725145217674.png)

문제는 그냥 컨테이너를 실행하면 컨테이너(가상환경)에서 한 작업물이 서버로 저장이 안된다. 그 문제를 해결하기위해 우선 `서버-컨테이너` 공용 저장소를 하나 만들어준다.

```
nvidia-docker run -it \
-p <서버 포트번호>:<도커 내부 포트번호> \
--name <컨테이너이름> \
-v <공용 저장소 폴더 주소>:/data \
<컨테이너 이미지>  bash
```

예를들면

```
nvidia-docker run -it \
-p 151515:10000 \
--name hello \
-v ~/share:/data \
pytorch/pytorch:latest  bash
```

이렇게 컨테이너를 만들어준다. 그러면 자동으로 컨테이너 쉘에 접속이 되는데 여기서 `jupyter notebook`를 설치해준다.

```
pip install ipykernel
pip install jupyter notebook
```

설치가 끝나면 jupyter notebook를 실행

```
jupyter notebook --ip 0.0.0.0 --port <도커 내부 포트번호> --allow-root --NotebookApp.token= --notebook-dir='/data'
```

예를들면 이렇게

```
jupyter notebook --ip 0.0.0.0 --port 10000 --allow-root --NotebookApp.token= --notebook-dir='/data'
```

그리고 웹창에서 

```
http://localhost:<로컬 포트번호>

Ex)
http://localhost:9999
```

로 접속하면 끝.

## 종료 및 다시켜기

컨테이너 종료시 쉘에

```
exit
```

를 치면 컨테이너가 종료상태가 된다. 다시 시작하려면 컨테이너를 켜주고 붙여주기만 하면됨.

```
docker start <컨테이너 이름>
docker attach <컨테이너 이름>
```

