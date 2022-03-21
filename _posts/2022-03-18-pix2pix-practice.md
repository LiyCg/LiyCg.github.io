---
title:  "Image-to-Image Translation with Conditional Adversarial Networks(CVPR2017))"
categories: DL_practice
tags:
  - Deep learning
  - DL
  - Generative Model
  - Generative Adversarial Networks
  - image translation
  - cGAN
  - domain change
toc: true
toc_sticky: true
toc_label: "Pix2Pix - Image-to-Image Translation with Conditional Adversarial Net"
toc_icon: "blog"
---

# Pix2Pix 코드 실습
 - 작성자: 이인엽(dldlsduq94@korea.ac.kr)
 - 논문제목: Image-to-Image Translation with Conditional Adversarial Networks(CVPR2017)
 - 대표적인 이미지간 도메인 변환 기술인 pix2pix 모델을 학습시켜보자
 - 학습 데이터셋: Facades(3X256X256)


## 트레이닝 데이터셋 불러오기
 - 학습을 위해 Facades 데이터셋 load
  - 데이터셋 크기는 매우 작지만, 좋은 결과 얻을 수 있음
  


```python
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```

    Cloning into 'pytorch-CycleGAN-and-pix2pix'...
    remote: Enumerating objects: 2447, done.[K
    remote: Total 2447 (delta 0), reused 0 (delta 0), pack-reused 2447[K
    Receiving objects: 100% (2447/2447), 8.18 MiB | 8.58 MiB/s, done.
    Resolving deltas: 100% (1535/1535), done.



```python
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')
```


```python
!pip install -r requirements.txt

```

    Requirement already satisfied: torch>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (1.10.0+cu111)
    Requirement already satisfied: torchvision>=0.5.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 2)) (0.11.1+cu111)
    Collecting dominate>=2.4.0
      Downloading dominate-2.6.0-py2.py3-none-any.whl (29 kB)
    Collecting visdom>=0.1.8.8
      Downloading visdom-0.1.8.9.tar.gz (676 kB)
    [K     |████████████████████████████████| 676 kB 7.9 MB/s 
    [?25hCollecting wandb
      Downloading wandb-0.12.11-py2.py3-none-any.whl (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 36.2 MB/s 
    [?25hRequirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.4.0->-r requirements.txt (line 1)) (3.10.0.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torchvision>=0.5.0->-r requirements.txt (line 2)) (1.21.5)
    Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision>=0.5.0->-r requirements.txt (line 2)) (7.1.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from visdom>=0.1.8.8->-r requirements.txt (line 4)) (1.4.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from visdom>=0.1.8.8->-r requirements.txt (line 4)) (2.23.0)
    Requirement already satisfied: tornado in /usr/local/lib/python3.7/dist-packages (from visdom>=0.1.8.8->-r requirements.txt (line 4)) (5.1.1)
    Requirement already satisfied: pyzmq in /usr/local/lib/python3.7/dist-packages (from visdom>=0.1.8.8->-r requirements.txt (line 4)) (22.3.0)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from visdom>=0.1.8.8->-r requirements.txt (line 4)) (1.15.0)
    Collecting jsonpatch
      Downloading jsonpatch-1.32-py2.py3-none-any.whl (12 kB)
    Collecting torchfile
      Downloading torchfile-0.1.0.tar.gz (5.2 kB)
    Collecting websocket-client
      Downloading websocket_client-1.3.1-py3-none-any.whl (54 kB)
    [K     |████████████████████████████████| 54 kB 2.4 MB/s 
    [?25hCollecting sentry-sdk>=1.0.0
      Downloading sentry_sdk-1.5.8-py2.py3-none-any.whl (144 kB)
    [K     |████████████████████████████████| 144 kB 46.0 MB/s 
    [?25hRequirement already satisfied: Click!=8.0.0,>=7.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (7.1.2)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (3.13)
    Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (2.8.2)
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (5.4.8)
    Collecting setproctitle
      Downloading setproctitle-1.2.2-cp37-cp37m-manylinux1_x86_64.whl (36 kB)
    Collecting shortuuid>=0.5.0
      Downloading shortuuid-1.0.8-py3-none-any.whl (9.5 kB)
    Requirement already satisfied: promise<3,>=2.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (2.3)
    Collecting docker-pycreds>=0.4.0
      Downloading docker_pycreds-0.4.0-py2.py3-none-any.whl (9.0 kB)
    Requirement already satisfied: protobuf>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (3.17.3)
    Collecting pathtools
      Downloading pathtools-0.1.2.tar.gz (11 kB)
    Collecting GitPython>=1.0.0
      Downloading GitPython-3.1.27-py3-none-any.whl (181 kB)
    [K     |████████████████████████████████| 181 kB 43.2 MB/s 
    [?25hCollecting yaspin>=1.0.0
      Downloading yaspin-2.1.0-py3-none-any.whl (18 kB)
    Collecting gitdb<5,>=4.0.1
      Downloading gitdb-4.0.9-py3-none-any.whl (63 kB)
    [K     |████████████████████████████████| 63 kB 1.8 MB/s 
    [?25hCollecting smmap<6,>=3.0.1
      Downloading smmap-5.0.0-py3-none-any.whl (24 kB)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (2021.10.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (2.10)
    Requirement already satisfied: termcolor<2.0.0,>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from yaspin>=1.0.0->wandb->-r requirements.txt (line 5)) (1.1.0)
    Collecting jsonpointer>=1.9
      Downloading jsonpointer-2.2-py2.py3-none-any.whl (7.5 kB)
    Building wheels for collected packages: visdom, pathtools, torchfile
      Building wheel for visdom (setup.py) ... [?25l[?25hdone
      Created wheel for visdom: filename=visdom-0.1.8.9-py3-none-any.whl size=655250 sha256=0e1903261a34333198000874ecc69509735c908dede660d8330da0b436047a2c
      Stored in directory: /root/.cache/pip/wheels/2d/d1/9b/cde923274eac9cbb6ff0d8c7c72fe30a3da9095a38fd50bbf1
      Building wheel for pathtools (setup.py) ... [?25l[?25hdone
      Created wheel for pathtools: filename=pathtools-0.1.2-py3-none-any.whl size=8806 sha256=e3f4ac3e14ea521630f3b8382e1f9a9eb7e5aa4ca490a4702997bfd65ccc5789
      Stored in directory: /root/.cache/pip/wheels/3e/31/09/fa59cef12cdcfecc627b3d24273699f390e71828921b2cbba2
      Building wheel for torchfile (setup.py) ... [?25l[?25hdone
      Created wheel for torchfile: filename=torchfile-0.1.0-py3-none-any.whl size=5709 sha256=4c78a2b4d1be5777a06aa37f207dd29ac50fc8de7df1c6695fa5bb08ca9476ab
      Stored in directory: /root/.cache/pip/wheels/ac/5c/3a/a80e1c65880945c71fd833408cd1e9a8cb7e2f8f37620bb75b
    Successfully built visdom pathtools torchfile
    Installing collected packages: smmap, jsonpointer, gitdb, yaspin, websocket-client, torchfile, shortuuid, setproctitle, sentry-sdk, pathtools, jsonpatch, GitPython, docker-pycreds, wandb, visdom, dominate
    Successfully installed GitPython-3.1.27 docker-pycreds-0.4.0 dominate-2.6.0 gitdb-4.0.9 jsonpatch-1.32 jsonpointer-2.2 pathtools-0.1.2 sentry-sdk-1.5.8 setproctitle-1.2.2 shortuuid-1.0.8 smmap-5.0.0 torchfile-0.1.0 visdom-0.1.8.9 wandb-0.12.11 websocket-client-1.3.1 yaspin-2.1.0



```python
!bash ./datasets/download_pix2pix_dataset.sh facades
```

    Specified [facades]
    WARNING: timestamping does nothing in combination with -O. See the manual
    for details.
    
    --2022-03-21 12:00:38--  http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
    Resolving efrosgans.eecs.berkeley.edu (efrosgans.eecs.berkeley.edu)... 128.32.244.190
    Connecting to efrosgans.eecs.berkeley.edu (efrosgans.eecs.berkeley.edu)|128.32.244.190|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 30168306 (29M) [application/x-gzip]
    Saving to: ‘./datasets/facades.tar.gz’
    
    ./datasets/facades. 100%[===================>]  28.77M  2.45MB/s    in 18s     
    
    2022-03-21 12:00:56 (1.57 MB/s) - ‘./datasets/facades.tar.gz’ saved [30168306/30168306]
    
    facades/
    facades/test/
    facades/test/27.jpg
    facades/test/5.jpg
    facades/test/72.jpg
    facades/test/1.jpg
    facades/test/10.jpg
    facades/test/100.jpg
    facades/test/101.jpg
    facades/test/102.jpg
    facades/test/103.jpg
    facades/test/104.jpg
    facades/test/105.jpg
    facades/test/106.jpg
    facades/test/11.jpg
    facades/test/12.jpg
    facades/test/13.jpg
    facades/test/14.jpg
    facades/test/15.jpg
    facades/test/16.jpg
    facades/test/17.jpg
    facades/test/18.jpg
    facades/test/19.jpg
    facades/test/2.jpg
    facades/test/20.jpg
    facades/test/21.jpg
    facades/test/22.jpg
    facades/test/23.jpg
    facades/test/24.jpg
    facades/test/25.jpg
    facades/test/26.jpg
    facades/test/50.jpg
    facades/test/51.jpg
    facades/test/52.jpg
    facades/test/53.jpg
    facades/test/54.jpg
    facades/test/55.jpg
    facades/test/56.jpg
    facades/test/57.jpg
    facades/test/58.jpg
    facades/test/59.jpg
    facades/test/6.jpg
    facades/test/60.jpg
    facades/test/61.jpg
    facades/test/62.jpg
    facades/test/63.jpg
    facades/test/64.jpg
    facades/test/65.jpg
    facades/test/66.jpg
    facades/test/67.jpg
    facades/test/68.jpg
    facades/test/69.jpg
    facades/test/7.jpg
    facades/test/70.jpg
    facades/test/71.jpg
    facades/test/73.jpg
    facades/test/74.jpg
    facades/test/75.jpg
    facades/test/76.jpg
    facades/test/77.jpg
    facades/test/78.jpg
    facades/test/79.jpg
    facades/test/8.jpg
    facades/test/80.jpg
    facades/test/81.jpg
    facades/test/82.jpg
    facades/test/83.jpg
    facades/test/84.jpg
    facades/test/85.jpg
    facades/test/86.jpg
    facades/test/87.jpg
    facades/test/88.jpg
    facades/test/89.jpg
    facades/test/9.jpg
    facades/test/90.jpg
    facades/test/91.jpg
    facades/test/92.jpg
    facades/test/93.jpg
    facades/test/94.jpg
    facades/test/95.jpg
    facades/test/96.jpg
    facades/test/97.jpg
    facades/test/98.jpg
    facades/test/99.jpg
    facades/test/28.jpg
    facades/test/29.jpg
    facades/test/3.jpg
    facades/test/30.jpg
    facades/test/31.jpg
    facades/test/32.jpg
    facades/test/33.jpg
    facades/test/34.jpg
    facades/test/35.jpg
    facades/test/36.jpg
    facades/test/37.jpg
    facades/test/38.jpg
    facades/test/39.jpg
    facades/test/4.jpg
    facades/test/40.jpg
    facades/test/41.jpg
    facades/test/42.jpg
    facades/test/43.jpg
    facades/test/44.jpg
    facades/test/45.jpg
    facades/test/46.jpg
    facades/test/47.jpg
    facades/test/48.jpg
    facades/test/49.jpg
    facades/train/
    facades/train/1.jpg
    facades/train/10.jpg
    facades/train/100.jpg
    facades/train/101.jpg
    facades/train/102.jpg
    facades/train/103.jpg
    facades/train/104.jpg
    facades/train/105.jpg
    facades/train/106.jpg
    facades/train/107.jpg
    facades/train/108.jpg
    facades/train/109.jpg
    facades/train/11.jpg
    facades/train/110.jpg
    facades/train/111.jpg
    facades/train/112.jpg
    facades/train/113.jpg
    facades/train/114.jpg
    facades/train/115.jpg
    facades/train/116.jpg
    facades/train/117.jpg
    facades/train/118.jpg
    facades/train/119.jpg
    facades/train/12.jpg
    facades/train/120.jpg
    facades/train/121.jpg
    facades/train/122.jpg
    facades/train/123.jpg
    facades/train/124.jpg
    facades/train/125.jpg
    facades/train/126.jpg
    facades/train/309.jpg
    facades/train/31.jpg
    facades/train/310.jpg
    facades/train/311.jpg
    facades/train/312.jpg
    facades/train/313.jpg
    facades/train/314.jpg
    facades/train/315.jpg
    facades/train/316.jpg
    facades/train/317.jpg
    facades/train/318.jpg
    facades/train/319.jpg
    facades/train/32.jpg
    facades/train/320.jpg
    facades/train/321.jpg
    facades/train/322.jpg
    facades/train/323.jpg
    facades/train/324.jpg
    facades/train/325.jpg
    facades/train/326.jpg
    facades/train/327.jpg
    facades/train/328.jpg
    facades/train/329.jpg
    facades/train/390.jpg
    facades/train/391.jpg
    facades/train/392.jpg
    facades/train/393.jpg
    facades/train/394.jpg
    facades/train/395.jpg
    facades/train/396.jpg
    facades/train/397.jpg
    facades/train/398.jpg
    facades/train/399.jpg
    facades/train/4.jpg
    facades/train/40.jpg
    facades/train/400.jpg
    facades/train/41.jpg
    facades/train/42.jpg
    facades/train/43.jpg
    facades/train/44.jpg
    facades/train/45.jpg
    facades/train/46.jpg
    facades/train/47.jpg
    facades/train/48.jpg
    facades/train/49.jpg
    facades/train/5.jpg
    facades/train/50.jpg
    facades/train/51.jpg
    facades/train/52.jpg
    facades/train/53.jpg
    facades/train/54.jpg
    facades/train/55.jpg
    facades/train/56.jpg
    facades/train/57.jpg
    facades/train/58.jpg
    facades/train/59.jpg
    facades/train/6.jpg
    facades/train/60.jpg
    facades/train/61.jpg
    facades/train/222.jpg
    facades/train/223.jpg
    facades/train/224.jpg
    facades/train/225.jpg
    facades/train/226.jpg
    facades/train/227.jpg
    facades/train/228.jpg
    facades/train/229.jpg
    facades/train/23.jpg
    facades/train/230.jpg
    facades/train/231.jpg
    facades/train/232.jpg
    facades/train/233.jpg
    facades/train/234.jpg
    facades/train/235.jpg
    facades/train/236.jpg
    facades/train/237.jpg
    facades/train/238.jpg
    facades/train/239.jpg
    facades/train/24.jpg
    facades/train/240.jpg
    facades/train/241.jpg
    facades/train/242.jpg
    facades/train/243.jpg
    facades/train/244.jpg
    facades/train/245.jpg
    facades/train/156.jpg
    facades/train/157.jpg
    facades/train/158.jpg
    facades/train/159.jpg
    facades/train/16.jpg
    facades/train/160.jpg
    facades/train/161.jpg
    facades/train/162.jpg
    facades/train/163.jpg
    facades/train/164.jpg
    facades/train/165.jpg
    facades/train/166.jpg
    facades/train/167.jpg
    facades/train/168.jpg
    facades/train/169.jpg
    facades/train/17.jpg
    facades/train/170.jpg
    facades/train/171.jpg
    facades/train/172.jpg
    facades/train/173.jpg
    facades/train/174.jpg
    facades/train/175.jpg
    facades/train/176.jpg
    facades/train/177.jpg
    facades/train/178.jpg
    facades/train/179.jpg
    facades/train/18.jpg
    facades/train/180.jpg
    facades/train/181.jpg
    facades/train/182.jpg
    facades/train/183.jpg
    facades/train/184.jpg
    facades/train/185.jpg
    facades/train/186.jpg
    facades/train/187.jpg
    facades/train/188.jpg
    facades/train/189.jpg
    facades/train/19.jpg
    facades/train/127.jpg
    facades/train/155.jpg
    facades/train/190.jpg
    facades/train/221.jpg
    facades/train/246.jpg
    facades/train/27.jpg
    facades/train/29.jpg
    facades/train/308.jpg
    facades/train/33.jpg
    facades/train/350.jpg
    facades/train/370.jpg
    facades/train/39.jpg
    facades/train/62.jpg
    facades/train/270.jpg
    facades/train/271.jpg
    facades/train/272.jpg
    facades/train/273.jpg
    facades/train/274.jpg
    facades/train/275.jpg
    facades/train/276.jpg
    facades/train/277.jpg
    facades/train/278.jpg
    facades/train/279.jpg
    facades/train/28.jpg
    facades/train/280.jpg
    facades/train/281.jpg
    facades/train/282.jpg
    facades/train/283.jpg
    facades/train/284.jpg
    facades/train/285.jpg
    facades/train/286.jpg
    facades/train/287.jpg
    facades/train/288.jpg
    facades/train/289.jpg
    facades/train/351.jpg
    facades/train/352.jpg
    facades/train/353.jpg
    facades/train/354.jpg
    facades/train/355.jpg
    facades/train/356.jpg
    facades/train/357.jpg
    facades/train/358.jpg
    facades/train/359.jpg
    facades/train/36.jpg
    facades/train/360.jpg
    facades/train/361.jpg
    facades/train/362.jpg
    facades/train/363.jpg
    facades/train/364.jpg
    facades/train/365.jpg
    facades/train/366.jpg
    facades/train/367.jpg
    facades/train/368.jpg
    facades/train/369.jpg
    facades/train/37.jpg
    facades/train/63.jpg
    facades/train/64.jpg
    facades/train/65.jpg
    facades/train/66.jpg
    facades/train/67.jpg
    facades/train/68.jpg
    facades/train/69.jpg
    facades/train/7.jpg
    facades/train/70.jpg
    facades/train/71.jpg
    facades/train/72.jpg
    facades/train/73.jpg
    facades/train/74.jpg
    facades/train/75.jpg
    facades/train/76.jpg
    facades/train/77.jpg
    facades/train/78.jpg
    facades/train/79.jpg
    facades/train/8.jpg
    facades/train/80.jpg
    facades/train/81.jpg
    facades/train/82.jpg
    facades/train/83.jpg
    facades/train/84.jpg
    facades/train/85.jpg
    facades/train/86.jpg
    facades/train/87.jpg
    facades/train/88.jpg
    facades/train/89.jpg
    facades/train/9.jpg
    facades/train/90.jpg
    facades/train/91.jpg
    facades/train/92.jpg
    facades/train/93.jpg
    facades/train/94.jpg
    facades/train/95.jpg
    facades/train/96.jpg
    facades/train/97.jpg
    facades/train/98.jpg
    facades/train/99.jpg
    facades/train/128.jpg
    facades/train/129.jpg
    facades/train/13.jpg
    facades/train/130.jpg
    facades/train/131.jpg
    facades/train/132.jpg
    facades/train/133.jpg
    facades/train/134.jpg
    facades/train/135.jpg
    facades/train/136.jpg
    facades/train/137.jpg
    facades/train/138.jpg
    facades/train/139.jpg
    facades/train/14.jpg
    facades/train/140.jpg
    facades/train/141.jpg
    facades/train/142.jpg
    facades/train/143.jpg
    facades/train/144.jpg
    facades/train/145.jpg
    facades/train/146.jpg
    facades/train/147.jpg
    facades/train/148.jpg
    facades/train/149.jpg
    facades/train/15.jpg
    facades/train/150.jpg
    facades/train/151.jpg
    facades/train/152.jpg
    facades/train/153.jpg
    facades/train/154.jpg
    facades/train/191.jpg
    facades/train/192.jpg
    facades/train/193.jpg
    facades/train/194.jpg
    facades/train/195.jpg
    facades/train/196.jpg
    facades/train/197.jpg
    facades/train/198.jpg
    facades/train/199.jpg
    facades/train/2.jpg
    facades/train/20.jpg
    facades/train/200.jpg
    facades/train/201.jpg
    facades/train/202.jpg
    facades/train/203.jpg
    facades/train/204.jpg
    facades/train/205.jpg
    facades/train/206.jpg
    facades/train/207.jpg
    facades/train/208.jpg
    facades/train/209.jpg
    facades/train/21.jpg
    facades/train/210.jpg
    facades/train/211.jpg
    facades/train/212.jpg
    facades/train/213.jpg
    facades/train/214.jpg
    facades/train/215.jpg
    facades/train/216.jpg
    facades/train/217.jpg
    facades/train/218.jpg
    facades/train/219.jpg
    facades/train/22.jpg
    facades/train/220.jpg
    facades/train/247.jpg
    facades/train/248.jpg
    facades/train/249.jpg
    facades/train/25.jpg
    facades/train/250.jpg
    facades/train/251.jpg
    facades/train/252.jpg
    facades/train/253.jpg
    facades/train/254.jpg
    facades/train/255.jpg
    facades/train/256.jpg
    facades/train/257.jpg
    facades/train/258.jpg
    facades/train/259.jpg
    facades/train/26.jpg
    facades/train/260.jpg
    facades/train/261.jpg
    facades/train/262.jpg
    facades/train/263.jpg
    facades/train/264.jpg
    facades/train/265.jpg
    facades/train/266.jpg
    facades/train/267.jpg
    facades/train/268.jpg
    facades/train/269.jpg
    facades/train/330.jpg
    facades/train/331.jpg
    facades/train/332.jpg
    facades/train/333.jpg
    facades/train/334.jpg
    facades/train/335.jpg
    facades/train/336.jpg
    facades/train/337.jpg
    facades/train/338.jpg
    facades/train/339.jpg
    facades/train/34.jpg
    facades/train/340.jpg
    facades/train/341.jpg
    facades/train/342.jpg
    facades/train/343.jpg
    facades/train/344.jpg
    facades/train/345.jpg
    facades/train/346.jpg
    facades/train/347.jpg
    facades/train/348.jpg
    facades/train/349.jpg
    facades/train/35.jpg
    facades/train/290.jpg
    facades/train/291.jpg
    facades/train/292.jpg
    facades/train/293.jpg
    facades/train/294.jpg
    facades/train/295.jpg
    facades/train/296.jpg
    facades/train/297.jpg
    facades/train/298.jpg
    facades/train/299.jpg
    facades/train/3.jpg
    facades/train/30.jpg
    facades/train/300.jpg
    facades/train/301.jpg
    facades/train/302.jpg
    facades/train/303.jpg
    facades/train/304.jpg
    facades/train/305.jpg
    facades/train/306.jpg
    facades/train/307.jpg
    facades/train/371.jpg
    facades/train/372.jpg
    facades/train/373.jpg
    facades/train/374.jpg
    facades/train/375.jpg
    facades/train/376.jpg
    facades/train/377.jpg
    facades/train/378.jpg
    facades/train/379.jpg
    facades/train/38.jpg
    facades/train/380.jpg
    facades/train/381.jpg
    facades/train/382.jpg
    facades/train/383.jpg
    facades/train/384.jpg
    facades/train/385.jpg
    facades/train/386.jpg
    facades/train/387.jpg
    facades/train/388.jpg
    facades/train/389.jpg
    facades/val/
    facades/val/30.jpg
    facades/val/50.jpg
    facades/val/73.jpg
    facades/val/1.jpg
    facades/val/10.jpg
    facades/val/100.jpg
    facades/val/11.jpg
    facades/val/12.jpg
    facades/val/13.jpg
    facades/val/14.jpg
    facades/val/15.jpg
    facades/val/16.jpg
    facades/val/17.jpg
    facades/val/18.jpg
    facades/val/19.jpg
    facades/val/2.jpg
    facades/val/20.jpg
    facades/val/21.jpg
    facades/val/22.jpg
    facades/val/23.jpg
    facades/val/24.jpg
    facades/val/25.jpg
    facades/val/26.jpg
    facades/val/27.jpg
    facades/val/28.jpg
    facades/val/29.jpg
    facades/val/3.jpg
    facades/val/51.jpg
    facades/val/52.jpg
    facades/val/53.jpg
    facades/val/54.jpg
    facades/val/55.jpg
    facades/val/56.jpg
    facades/val/57.jpg
    facades/val/58.jpg
    facades/val/59.jpg
    facades/val/6.jpg
    facades/val/60.jpg
    facades/val/61.jpg
    facades/val/62.jpg
    facades/val/63.jpg
    facades/val/64.jpg
    facades/val/65.jpg
    facades/val/66.jpg
    facades/val/67.jpg
    facades/val/68.jpg
    facades/val/69.jpg
    facades/val/7.jpg
    facades/val/70.jpg
    facades/val/71.jpg
    facades/val/72.jpg
    facades/val/74.jpg
    facades/val/75.jpg
    facades/val/76.jpg
    facades/val/77.jpg
    facades/val/78.jpg
    facades/val/79.jpg
    facades/val/8.jpg
    facades/val/80.jpg
    facades/val/81.jpg
    facades/val/82.jpg
    facades/val/83.jpg
    facades/val/84.jpg
    facades/val/85.jpg
    facades/val/86.jpg
    facades/val/87.jpg
    facades/val/88.jpg
    facades/val/89.jpg
    facades/val/9.jpg
    facades/val/90.jpg
    facades/val/91.jpg
    facades/val/92.jpg
    facades/val/93.jpg
    facades/val/94.jpg
    facades/val/95.jpg
    facades/val/96.jpg
    facades/val/97.jpg
    facades/val/98.jpg
    facades/val/99.jpg
    facades/val/31.jpg
    facades/val/32.jpg
    facades/val/33.jpg
    facades/val/34.jpg
    facades/val/35.jpg
    facades/val/36.jpg
    facades/val/37.jpg
    facades/val/38.jpg
    facades/val/39.jpg
    facades/val/4.jpg
    facades/val/40.jpg
    facades/val/41.jpg
    facades/val/42.jpg
    facades/val/43.jpg
    facades/val/44.jpg
    facades/val/45.jpg
    facades/val/46.jpg
    facades/val/47.jpg
    facades/val/48.jpg
    facades/val/49.jpg
    facades/val/5.jpg


## 필요 라이브러리 불러오기
 - pytorch 라이브러리 load


```python
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

```

 - 학습 데이터셋 출력해 보기
  - 각 학습 이미지는 (256X256) 크기의 이미지 2개를 이어 붙인 형태를 가진다(paired)



```python
print("학습 데이터셋 A와 B의 개수:", len(next(os.walk('/content/facades/train/'))[2]))
print("평가 데이터셋 A와 B의 개수:", len(next(os.walk('/content/facades/val/'))[2]))
print("테스트 데이터셋 A와 B의 개수:", len(next(os.walk('/content/facades/test/'))[2]))
```

    학습 데이터셋 A와 B의 개수: 400
    평가 데이터셋 A와 B의 개수: 100
    테스트 데이터셋 A와 B의 개수: 106



```python
# 한쌍만 출력해보자(왼쪽은 정답/target, 오른쪽은 조건/condition)
image = Image.open('/content/facades/train/9.jpg')
print("이미지 크기: ", image.size)

plt.imshow(image)
plt.show()

```

    이미지 크기:  (512, 256)



    
![png](output_10_1.png)
    


 - Custom Dataset 클래스 정의


```python
# inherit from 'Dataset' library
class ImageDataset(Dataset):
  # 데이터 읽기
  def __init__(self,root,transforms_ =None, mode = "train"):
    self.transform = transforms_
    
    # glob.glob():
    # 많은 파일들을 다뤄야 하는 파이썬 프로그램을 작성할 때, 특정한 패턴이나 확장자를 가진 파일들의 경로나 이름이 필요할 때가 있다. 
    # glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다. 
    # 단, 조건에 정규식을 사용할 수 없으며 엑셀 등에서도 사용할 수 있는 '*'와 '?'같은 와일드카드만을 지원한다.
    # sorted():  
    # 또 sorted() 라는 내장 함수는 이터러블 객체로부터 정렬된 리스트를 생성한다.
    self.files = sorted(glob.glob(os.path.join(root,mode) + "/*.jpg"))
    
    # 데이터 갯수가 적기 때문에 테스트 데이터를 학습시기에 사용 
    if mode == "train":
      self.files.extend(sorted(glob.glob(os.path.join(root,"test") + "/*.jpg")))
  
  # index번째 샘플을 찾는데 사용된다. 이미지판독
  def __getitem__(self, index):
    img = Image.open(self.files[index % len(self.files)])
    w, h = img.size
    img_A = img.crop((0,0,w/2,h))
    img_B = img.crop((w/2,0,w,h))

    # Data augmentation을 위해 좌우반전(horizontal flips)
    if np.random.random() < 0.5:
      # Image.fromarray()함수를 사용하여 배열을 PIL 이미지 객체로 다시 변환
      img_A = Image.fromarray(np.array(img_A)[:,::-1,:], "RGB")
      img_B = Image.fromarray(np.array(img_B)[:,::-1,:], "RGB")
    
    img_A = self.transform(img_A)
    img_B = self.transform(img_B)

    return {"A": img_A, "B": img_B}

  def __len__(self):
    return len(self.files)


```


```python
# 이미지 변형(전처리)
transforms_ = transforms.Compose([
                                  # 이미지 사이즈를 size로 변경한다
                                  transforms.Resize((256,256), Image.BICUBIC),
                                  # transforms.ToTensor() - 이미지 데이터를 tensor로 바꿔준다. / numpy 이미지에서 torch 이미지로 변경
                                  transforms.ToTensor(),
                                  # transforms.Normalize(mean, std, inplace=False) - 이미지를 정규화한다.(정규분포로)
                                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# 정의한 custom class로 데이터 관리 
train_dataset = ImageDataset("/content/facades", transforms_= transforms_)
val_dataset = ImageDataset("/content/facades", transforms_= transforms_)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=4)

```

    /usr/local/lib/python3.7/dist-packages/torchvision/transforms/transforms.py:288: UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
      "Argument interpolation should be of type InterpolationMode instead of int. "
    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))


## 생성자 및 판별자 모델 정의 
 - pix2pix는 cGAN의 구조를 가지며, 이미지를 조건(condition)으로 입력 받아 이미지를 출력한다. 
  - 입력차원과 출력차원이 동일한 아키텍쳐로 U-Net을 사용한다. 
  - U-Net의 skip connection을 이용한다. 
  - 많은 low-level information이 입력과 출력 과정에서 공유될 수 있다. 
  ![download.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAC7QAAAL8CAYAAABqeBheAAAgAElEQVR4AezdB5QUVf7/fXMWRUAB1xxRMQvmtKhrWBHDYlpd0TXLqmvCvAbMESMmFHNAMbPmNaFiAHTNAgpmUcEcfvc577v/mqe6p6dzz0zPvOucPpO6K7xuVYn3fup7ZwguCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgq0gMAMLbBNN6mAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiigQDDQ7kmggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoo0CICBtpbhN2NKqCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACChho9xxQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUaBEBA+0twu5GFVBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBQy0ew4ooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKtIiAgfYWYXejCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgbaPQcUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFWkTAQHuLsLtRBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQPtngMKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACLSJgoL1F2N2oAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggIF2zwEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUECBFhEw0N4i7G5UAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQwEC754ACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKBAiwgYaG8RdjeqgAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgooYKDdc0ABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUKBNCEyePDnce++94aabbgojRowIDz/8cHjyySfD008/XfDF+0aNGhXuvvvucPPNN8fPT5kypaDLxIkT42eGDx8e7rjjjrj9hx56KDzyyCPh8ccfD0899VR44oknwqOPPhr35/777w933nlnuPHGG8MNN9wQPvjgg4LbqPQNP/30U9z2ZZddFop5DR06NLz88svhxx9/rGjTHPuwYcPCLbfcEo0eeOCB8O9//7vBhb9jhDt/w573XnfddWHMmDEVb7+inffDCiiggAIKKKCAAgoooIACCiiggALNJmCgvdmo3ZACCiiggAKZAv/3f/8Xvv322zBp0qTw2muvNQysff/995lv9KdWKfD777+Hb775JkyYMCG88sorceDtpZdeCj/88EOr3F93qroCv/32W5g6dWp4//334+AuA9Ljxo1zkLW6zK5NAQUUUEABBRRQQAEFFFBAgZIFCKX36dMndO7cOXTp0iUstNBCoWvXrqFbt24FX7yP9y+44ILx8+uvv34MoxfaCULzG2+8ccM2+TzrSbbNepNXsn72jX1cbLHFwj333FNoExX/nX6MPffcMyywwAJFv1ZdddVA+J5+kHKXQw45JLYDx1rIJbHhvZ06dQpnnHFG+PLLL8vdtJ9TQAEFFFBAAQUUUEABBRRQQAEFFKgjAQPtddRY7qoCClQm8Pbbb4err746DBkypNW8qO5DKNalfQhQXWivvfYKm2++eVhrrbXCsssuGxZffPGwyCKLhIUXXjgOam255Zbhww8/bB8gdXaUVIXafffd44DoGmusEZZZZpmG9uvevXtsvwEDBoRiqnbV2aG7uyGEiy66KPTv3z9suummYbXVVgtLL710HHDm+k3a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A8F2Cp60xiUdaH/nnXditforrrgiLL300hmh9plmmilWVX/mmWcqPgwqmhNo33rrrcPLL79c8fpaYgW333576Nmzp4H2lsB3mwoooIACCiiggAIKKKCAAgoo0OYEDLS3uSb1gBRQoByB5gy0/9///V+47LLL4iDGPPPME959991ydtnPtBEBA+313ZAG2uu7/SrdewPtlQr6eQUUUEABBRRQQAEFFFBAAQWaV8BAe27v7EA775o6dWp8eH/eeefNCLXPNtts4ZBDDgkffPBB7pUV+du2EGg/66yzAlXr+/XrFzgeFwUUUEABBRRQQAEFFFBAAQUUUECB8gUMtJdv5ycVUKANCTRnoB22L7/8Miy//PKBijajR48Ov/32WxvS9FBKETDQXopW63uvgfbW1ybNuUcG2ptT220poIACCiiggAIKKKCAAgooULmAgfbchrkC7RRmGTt2bNhss81iP3YyQyFfO3XqFK688srw3Xff5V5hEb+t90D777//Hg444IBAwN9AexEN7lsUUEABBRRQQAEFFFBAAQUUUECBAgIG2gsA+WcFFGgfAs0daGcwoH///mHOOecMI0aMCD/++GP7gPYoGwkYaG9EUle/MNBeV81V9Z010F51UleogAIKKKCAAgoooIACCiigQE0FDLTn5s0VaOedv/76a7jhhhtCjx49Mqq0zzjjjGGNNdYIjz/+eKCvu5yl3gPt48ePD5tvvnl0MdBezhngZxRQQAEFFFBAAQUUUEABBRRQQIFMAQPtmR7+pIAC7VSguQPtMF944YWxks2QIUPC9OnT26m8h22gvb7PAQPt9d1+le69gfZKBf28AgoooIACCiiggAIKKKCAAs0rYKA9t3dTgXbeTd/1EUccETp27JgRap911lnDgAEDwttvv517pQV+W8+BdkL8nEtdu3Y10F6gnf2zAgoooIACCiiggAIKKKCAAgooUKyAgfZipXyfAgq0aYGWCLS/+OKLYZFFFglHH310mDp1apv29eCaFjDQ3rRNPfzFQHs9tFLt9tFAe+1sXbMCCiiggAIKKKCAAgoooIACtRAw0J5bNV+gnU8QWt9mm23CzDPPnBFqn2+++WLhlm+++Sb3ivP8tp4D7a+//nrYaqutwkwzzWSgPU8b+ycFFFBAAQUUUEABBRRQQAEFFFCgFAED7aVo+V4FFGizAtUKtFOZ5dtvvw0//vhjwalWed9SSy0Vdtttt/DZZ5+1WVsPLL+Agfb8Pq39rwbaW3sL1Xb/DLTX1te1K6CAAgoooIACCiiggAIKKFBtAQPtuUULBdp/++23cNddd4VVVlklI9A+wwwzhBVXXDE8+OCD4ffff8+98iZ+W6+B9q+++ir84x//CPPPP3+DRb9+/QLH46KAAgoooIACCiiggAIKKKCAAgooUL6Agfby7fykAgq0IYFqBdo//vjjsMEGG4R//etf4euvv84rRPi9V69eYdNNNw2TJ0/O+17/2HYFDLTXd9saaK/v9qt07w20Vyro5xVQQAEFFFBAAQUUUEABBRRoXgED7bm9CwXa+dT3338fTjzxxLDgggs2BLkJtM8yyyxh5513DuPHj8+98iZ+W8tAO33vU6ZMCY8//ni47bbbwi233BLuv//+8Oabb4YffvihiT0q/OtPP/00HHvssaF79+5hxhlnbHAw0F7YzncooIACCiiggAIKKKCAAgoooIAChQQMtBcS8u8KKNAuBKoVaJ84cWJYfPHFY4WWL7/8sqAd211//fXDhx9+WPC9vqFtChhor+92NdBe3+1X6d4baK9U0M8roIACCiiggAIKKKCAAgoo0LwCBtpzexcTaOeT9H/vuOOOYdZZZ20IcxNqn3vuucMZZ5wRiukTT/agFoH2CRMmhPPPPz/06dMnzo7arVu30KVLl/haaKGFwiKLLBJWXXXVMHDgwDB69Ojw008/JbuT9+sXX3wRRowYEf785z+Hjh07hplmminj+Nddd91w6aWXhoceeqjJ13/+85/AelwUUEABBRRQQAEFFFBAAQUUUEABBXILGGjP7eJvFVCgnQlUK9D+0ksvxU5xphwtpvP+s88+i9XZf/3116LFqS4zbdq0GIJ/9913w88//9zkZ6dOnRqrzowZMyaMHTs27lOpU782ufIm/sD+ffDBB+Gee+6JgxjHHHNMGDRoUDj11FNjNRwGPZiitpoLBq+//nq48cYbwymnnBKOPvrouM0LLrggPPLII+Hzzz8vecrbSvePtqUq0csvvxz++9//xso/2GQvtQy0UzGI4z/vvPOiBy74UJHojTfeyHvuZO9nUz9z7jIQ8/bbb8fzK1fbfvfdd4Fz9dVXXy35fG9qu7X8Pe3EbAvjxo2L7ffOO+80ObhVq0A7+/DRRx/FylFnnXVW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AgoooIACCiiggAKVCxhor9zQNSigQBsQaKlA+0033RQr9RKkpmO/0NKSgXb2jYBqhw4dMgYedt1116L2vdCx5fr7f//738D0s+mBDqaQpZJ3pcuBBx4Y5p133ox1n3baaeHrr7/OuWqmwJ1//vkz3s86Pv/885zvL/aX5QTaGbihUv2ss86asT8333xz+OGHH4rddMXvq5dAOwe69957N6rofeqpp+YMQJcCU06gncG11VdfPaPi/4wzzhgef/zxmj2QQBX2XA+knHvuuU2e86U45HovlbEWXnjhjHOUAcpqPJCy/vrrxwHP5N6A33333ZdzwJFj79u3b6MAPLNLVDLQWW6gfeLEiY0eLmBAt9KK8bnaIN/vDLTn0/FvCiiggAIKKKCAAgoooIACbVmAvgn6SehPSPoW+H/za665JlYUr9ax07e5xRZbhM6dOwcKd1B0obUutQy004fy7LPPhg022KDBO3HnaymBdvzY1169emWsi/79Smc/pM9mwIABgaIJn332Wd6m4n3p84diKK+99lrezxDEp+gCxQ5cFFBAAQUUUEABBRRQQAEFFFBAAQUaCxhob2zibxRQoB0KtFSgnTDzXHPNFahy3lSQOt0cLR1of/rpp8Mf/vCHjMECqj5XI6DKdLGEsRngSJZffvklTiGbrsTMdMCvvPJKIKRayfKnP/0po9r83HPPHQdWmhpYeu6558Jiiy2WcewMtkyePLmS3YjVidh2Moiz9tprF+V5/vnnx8Gw5HN8PfHEEysOaHMwVIomqE9wPt9ST4H222+/vVG4+u9//3v45JNP8h1iwb+VE2hPqnNlPyBx+eWXVxSwTnaWilZUJU9fS/ztwgsvDDwQkj5nqlExnWuR65cppNML1wZTaqcH9/j57bffTr+t5O9pM2ZqSB8H633jjTcaHXOy8iFDhjSa7eH4448PVK4vd8GX2TXS+8FDLk3N8pBsh3sMVbzmmWeehs9idOeddwamPa9koS0IxjNzRHb7Z6/XQHu2iD8roIACCiiggAIKKKCAAgq0FwH6vHjIP7ufc8899wxvvfVWVRjYximnnBL7YnbfffeKw9ZV2ak8K6lloJ3NUvX80ksvbVQZn36VUgPt06dPDwcccEDsV0/6ZXr06BFGjRqV5wgL/4niMyussEI4/fTTC/ax7rfffoF+6mT7xQTajzjiiMBMAA88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QAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUECBLAED7Vkg/qiAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCijQPAIG2pvH2a0ooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKZAkYaM8C8UcFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUECB5hEw0N48zm5FAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQIEvAQHsWiD8qoIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKNI+AgfbmcXYrCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAlkCBtqzQPxRAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQoHkEDLQ3j7NbUUABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFMgSMNCeBeKPCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAs0jYKC9eZzdigIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooECWgIH2LBB/VEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFGgeAQPtzePsVhRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQUUUEABBRRQQAEFFFBAAQWyBAy0Z4H4owIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIKKKCAAgoooEDzCBhobx5nt6KAAgoooIACCiiggAIKKKCAAgoooIACCiiggAIK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diVAgAABAgQIECBAYCogaDcJoxEQtJcd3AjaY0Zd0B4zl4L2mHm1K4F5CQjaY+QF7fmugvZ8u647Be1dOq4RIECAAAECfQQE7fXPWQTtfSavfY2gvf5MCtrb582V8Qrs3LkzNe/W/q53vSs9/vGPT/vvv/8twvbmc0cffXR68YtfnC644IK0a9euom9Y0F7EN4qbBe1lr0GC9pgxF7THuNqVAAECBAgQIECAAIGpgKDdJIxGQNBednAjaI8ZdUF7zFwK2mPm1a4E5iUgaI+RF7Tnuwra8+267hS0d+m4RoAAAQIECPQRELTXP2cRtPeZvPY1gvb6Mylob583V8YvcN1116Xzzz8/veY1r0lHHXVU2rp165qwfcuWLenggw9Oxx13XHrrW9+arrrqquxvWtCeTTeaGwXtZa9BgvaYURe0x7jalQABAgQIECBAgACBqYCg3SSMRkDQXnZwI2iPGXVBe8xcCtpj5tWuBOYlIGiPkRe057sK2vPtuu4UtHfpuEaAAAECBAj0ERC01z9nEbT3mbz2NYL2+jMpaG+fN1eWR+Dqq69OZ599dnrOc56TDj300DVR+6ZNm9K2bdvSPe95z7R9+/Z01llnpeuvv37wNy9oH0w2uhsE7WWvQYL2mJEXtMe42pUAAQIECBAgQIAAgamAoN0kjEZA0F52cCNojxl1QXvMXAraY+bVrgTmJSBoj5EXtOe7Ctrz7bruFLR36bhGgAABAgQI9BEQtNc/ZxG095m89jWC9vozKWhvnzdXlktg586d6ZJLLknvete70uMf//i0//773yJsbz539NFHpxe/+MXpggsuSLt27eqNIGjvTTXahYL2stcgQXvM6AvaY1ztSoAAAQIECBAgQIDAVEDQbhJGIyBoLzu4EbTHjLqgPWYuBe0x82pXAvMSELTHyAva810F7fl2XXcK2rt0XCNAgAABAgT6CAja65+zCNr7TF77GkF7/ZkUtLfPmyvLKXDdddel888/P73mNa9JRx11VNq6deuasH3Lli3p4IMPTscdd1x661vfmq666qpeEIL2XkyjXiRoL3sNErTHjL+gPcbVrgQIECBAgAABAgQITAUE7SZhNAKC9rKDG0F7zKgL2mPmUtAeM692JTAvAUF7jLygPd9V0J5v13WnoL1LxzUCBAgQIECgj4Cgvf45i6C9z+S1rxG0159JQXv7vLmy3AJXX311Ovvss9NznvOcdOihh66J2jdt2pS2bduW7nnPe6bt27ens846K11//fWdIIL2Tp6luChoL3sNErTHPA0E7TGudiVAgAABAgQIECBAYCogaDcJoxEQtJcd3AjaY0Zd0B4zl4L2mHm1K4F5CQjaY+QF7fmugvZ8u647Be1dOq4RIECAAAECfQQE7fXPWQTtfSavfY2gvf5MdgXtP/nJT9IXvvCF9JGPfGT287vf/W666aabZg/SpZdemj796U/PrjeP0eWXXz673nzwzW9+M3384x+frTnnnHNSE/+u/PjpT3+a/uu//mt2vfn9LrjggnTjjTeuLJm8S/ZnPvOZ2ZrTTz89/eAHP5hdbz74zne+kz7xiU/M1jTrm2h59Y9zzz03ffSjH52t+epXv5qa73PlR/N1ff7zn59db76Wiy66KO3atWtlSbrkkkvSpz71qdma5uMrr7xydr354Otf/3r62Mc+NlvzpS99Kf34xz+erWni6OZ/s622/fa3v50ai5UfjWMTXK+saQwvu+yylcuTX5t7zjzzzNma5mu/5pprZmt27tyZmt97ZY/m1+bdyVfH2Y3RZz/72TVrmu9x9Y/G4JOf/ORsTfN17f7O5eedd94a28b62muvnW2zY8eO1Dz2q7+WZp6ar3HlR/OYrp6n5vfcfZ6+8Y1vpDPOOGO2T+O4ep6aefu3f/u39KY3vSm9973vna1b/fuufPzBD34wvfnNb05Pf/rT143am7B9//33T/e73/3Si170oslcrp6Fla+7+VXQvlpjOT8WtJe9BgnaY54XgvYYV7sSIECAAAECBAgQIDAVELSbhNEICNrLDm4E7TGjLmiPmUtBe8y82pXAvAQE7THygvZ8V0F7vl3XnYL2Lh3XCBAgQIAAgT4Cgvb65yyC9j6T175G0F5/JruC9u9973vp5JNPTkccccTs52mnnbYmUm7C8sc//vGz6494xCMm4fPqR/HVr351evCDHzxb84d/+IeT4HtlTRMCN2cVq3+f1772tWuC6c997nPpyU9+8mzN/e9//0movLJH82sTMf/ar/3abE3zZ1gTyq/+8ad/+qfpPve5z2zN//k//yddfPHFsyVNLH3SSSfNrjdf09vf/vY1oXkTSf/mb/7mbM1xxx03ibBnm6SUmsDw6KOPnq35kz/5k0kMvbKm+Q8BnvGMZ8yuN79PE2CvDrObmPtJT3rSbM2v/MqvTELulT2aX9/whjekhz3sYbM1T33qU1MTkq/8uO6669Lznve82fXm93n5y1++Joz/4he/mE488cTZmsbn/e9//8oWk1//4z/+Ix177LGzNc3X1Twmq3/81V/91ST8Xnkc/+zP/iw1wfrKj+Y/ONh9nt7ylresieub0Hz1PD360Y+e/EcKK3s0v77qVa9KxxxzzOxr+aM/+qM189SYHHXUUenwww9Pv/ALvzBbt/J1rffr3e9+97Tffvvd4l3am6B95eed7nSnyWys9l39dQnaV2ss58eC9rLXIEF7zPNC0B7jalcCBAgQIECAAAECBKYCgnaTMBoBQXvZwY2gPWbUBe0xcyloj5lXuxKYl4CgPUZe0J7vKmjPt+u6U9DepeMaAQIECBAg0EdA0F7/nEXQ3mfy2tcI2uvPZFfQ3rz792/8xm/Mgt4m7G1i4tXvNP7Od74z3fe+952tuetd75qad71e/aMJmw888MDZmsc85jGpiahXfjTvEP5bv/Vbs+vN7/PCF75wTXTdvPP6gx70oNmabdu2pebvKFb/+Lu/+7s177LdrG/e7Xv1jybc3rx582yfJq5vQuuVH1/+8pcnUfxKxNz8+sY3vnHNu8U34fzP//zPz/a4973vnZqwf/WPZz/72em2t73tbM1v//Zvp+YdzFd+NP+xwGMf+9jZ9eb3+eu//us17yjfODZh9srXcoc73CG9+93vXtli8uupp56a7nznO8/WPOpRj1oTmjfvPt/8Wb6yR/Prc5/73Mm7zK9s1ES6TRS/sqbxeetb37pyefLr61//+tRE3ytrmq+reUxW//j93//9tPfee8/WNP9hQPNu6Ss/mneG332emv/YYfU75DeP6ZFHHjnb47DDDksf+MAHVraY/NrM0+1vf/vZmsc97nFr5umlL33pGpOVr7nGr83v+6//+q9r5mHlixO0r0gs76+C9rLXIEF7zHND0B7jalcCBAgQIECAAAECBKYCgnaTMBoBQXvZwY2gPWbUBe0xcyloj5lXuxKYl4CgPUZe0J7vKmjPt+u6U9DepeMaAQIECBAg0EdA0F7/nEXQ3mfy2tcI2uvPpKBd0C5o/7/vwr6n6H2fffaZvJP+jTfeeIs/qATttyBZuk8I2stegwTtMU8JQXuMq10JECBAgAABAgQIEJgKCNpN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b968Jmbftm1bevSjH51OO+20dOGFF6Ybb7wx6xsXtGexjeomQXvZa5CgPWbcBe0xrnYlQIAAAQIECBAgQGAqIGg3CaMRELSXHdwI2mNGXdAeM5eC9ph5tSuBeQkI2mPkBe35roL2fLuuOwXtXTquESBAgAABAn0EBO31z1kE7X0mr32NoL3+TAra2+fNlXELXHvttek973lPOv7449MhhxyStm7duiZk37RpUzriiCPSqaeempr//bxjx4508803Z3/TgvZsutHcKGgvew0StMeMuqA9xtWuBAgQIECAAAECBAhMBQTtJmE0AoL2soMbQXvMqAvaY+ZS0B4zr3YlMC8BQXuMvKA931XQnm/XdaegvUvHNQIECBAgQKCPgKC9/jmLoL3P5LWvEbTXn0lBe/u8uTJOgV27dqVzzjknnXLKKenII49M++677y1C9oMOOiht3749nX766emKK65IO3fuLP5mBe3FhAu/gaC97DVI0B4z4oL2GFe7EiBAgAABAgQIECAwFRC0m4TRCAjayw5uBO0xoy5oj5lLQXvMvNqVwLwEBO0x8oL2fFdBe75d152C9i4d1wgQIECAAIE+AoL2+ucsgvY+k9e+RtBefyYF7e3z5sr4BC655JL0qle9Kj384Q9PBx54YNq8efOamH3btm3p2GOPTaeddlr6zne+k2688cZq36SgvRrlwm4kaC97DRK0x4y2oD3G1a4ECBAgQIAAAQIECEwFBO0mYTQCgvaygxtBe8yoC9pj5lLQHjOvdiUwLwFBe4y8oD3fVdCeb9d1p6C9S8c1AgQIECBAoI+AoL3+OYugvc/kta8RtNefSUF7+7y5Mj6Bj370o+lhD3tY2rp165qQfdOmTemII45Ip556amr+t/KOHTvSzTffXPUbFLRX5VzIzQTtZa9BgvaYsRa0x7jalQABAgQIECBAgACBqYCg3SSMRkDQXnZwI2iPGXVBe8xcCtpj5tWuBOYlIGiPkRe057sK2vPtuu4UtHfpuEaAAAECBAj0ERC01z9nEbT3mbz2NYL2+jMpaG+fN1fGJ3DxxRenE044Ie2zzz6zoP2ggw5K27dvT6effnq64oor0s6dO0O+MUF7COtCbSpoL3sNErTHjLOgPcbVrgQIECBAgAABAgQITAUE7SZhNAKC9rKDG0F7zKgL2mPmUtAeM692JTAvAUF7jLyg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JiYmJjsh0J53LkppTaC9lJnQDwIECBAgQIAAAQLjKSDQPp7zOpajEmjPf0FLoH0sl4pBESBAgAABAgcREGg/CNBhvizQfphw/owAAQIECBCoBQTa85//GxZoP+uss9JnP/vZtHfv3to+ftm1a1e644470pe+9KXJn6997Wvp/vvvr7eJgFqE2qrX4z7Cag888EC9zX333Ze2bdtWbxPtxXPN2//93/+lrVu31tt89atfTfv27as3efDBB1Psu7mfb37zm3VALjb8zne+k7Zv3z6wzb333lu3Eb/EN8vGe/+qnTvvvHPKmL/xjW+kL3/5y/U28fi73/1u3U6MP/pXtRH30W4E/Kpb7Dfarrb5yle+kr797W9XL0/eR/+btjt37hwYcxjGc1UbcR9OTdtos2kb/W5/G+7dd9+dYv9VO2EUVs3b17/+9fr12C7mvTnmmIv2mO+5554U81Ld4nHTNn5vj7ldT7HfZj3FPofVU9O2XU9RNwerp2izWU/RXrue7rrrroF6irVwqPUUc9FeQ9Opp/aYo56atu16GmbbrqcYX3PMsVYPtZ6iPtv1FOu7WU9RF816in4Pq6cqzBr1Etsfaj3FehpWT821Op16ivXQrKcYX3uttuspaqN5fGrXU6zJYfXUXKvteop+t49Pw9bqsHqq1nLct49PMeeHU0/tMQ+rp+ZaPVA9NW2HrdVh9dS0bddTtNeup2HH/mH1VB2b4r597I+1eqj1FONvr6GYs+ZanW49HezYf7B6ir43j0/R3qHWU/Q76qdZT3GMjn9nmmui+Xr79//93/9NV155ZXr1q1+dHv/4x08JtUewfcmSJemXfumX0j/8wz9M9jnaX7Vq1cC2f/iHr0y33PrFFNcnu/55/RsuTI9+9GPq/Z900knpox/9aLNc/H4YAgLth4HmTwgQIECAAAECBAgQmLaAQPu0qWw40wIC7d2c3LnggtenRzziEfUJneOPP37yBNZMz7f9EyBAgAABAgS6EohAQHxzVFxsq37e+c53TglGdLX/cW1XoH1cZ9a4CBAgQIDA0RMQaM9//m9YoP2xj31set7znpduvPHGgcn9p3/6p/T0pz89/cqv/Mrkz1/+5V9OhlmrjSLo+ad/+qf167Hde97znoFg3ac//en0kpe8pN7mtNNOS/Fc8xbvvZ/97GfX27zqVa9Kt912W71JhCf//M//vH499vPmN795MnhdbRTBugjmV32N+wipNQOWH/zgB9Nv/uZv1tu8+MUvTp///OerJibvL7jggvTUpz613ubv/u7vJkN41UYRIvyTP/mT+vXYz7//+78PBJljv7/zO79Tb/P85z8/3XDDDVUTk/cXX3xx+o3f+I16m9e+9rWTwcJqowiIxzfbNsfzjne8I0WAsrrFhxBe9rKX1duccsopadOmTdXLk/fve9/7Jue2aucP/uAP0pYtWwa2Wbt2bd1GbBfzHoHK6nb77benP/qjPxrYJkKMzZDyf/zHf6Tf/u3frreJb+2N/jVv0e6pp55abxP7jXB0dYvgarue3vve9w6Eqj/1qU8N1NOznvWsFM81b+EUz1djPuecc9Ktt95abxIh4b/4i7+oX4/t1q1bl6Keq1t8o3Cznp7ylKekj3zkI9XLk/cf+MAHBurpzDPPnFJPUT+/9mu/Vu8r6itqqLrF+M8+++z69ehL1FMzjH7NNdek1atX19u88IUvTNdff33VxOR9rIdYW9WYY71EaLq6RYj03HPPrV+P7WLdRdC3ukWN/t7v/V69Taz9j33sY9XLk/exvp/73OfW27zyla8cqKeoib/5m7+pX4/9XHTRRZOh5KqhmIth9dRcqx/+8IdTHP+r8bzoRS+aUk//+I//mKLmq23+9m//dvJDFdV+IuT8Z3/2Z/Xrsd373//+gSBzrJc4DlRtxNja9fT2t789nX766fU2UaPNeoogcRwbqzbi/i1vectkMLjqy+c+97kUa6/aJvod89q8/du//Vs644wz6m1e/vKXD62nqo24f8Mb3jAZYK/aiTlv11Mc95ofULn66qsH6inW7XXXXVc1MXkf9fS0pz2t7svrXve6gXqKIPR5551Xvx59ufTSSweO/dFms56e+cxnTqmnd7/73VOO/XEsr26xVmNem2N+4xvfOOXY36ynX/3VX01xPGre4pvEf+u3fqtuJ/5N+sxnPtPcJEU9Ndfq+eefP3BdLELer3nNa+o2ok9RT80PdMR6adbTC17wgvSJT3xiYD9ve9vb0jOe8Yy6nTj2N/+9iw+N/NVf/VX9euznX/7lXwaO/XGupXl8irV61VVXDeznsssuGzj2v+IVrxiop1irr3/96wf28/d///fpkksumQycN80P9vvP/uzPph/8wR+sz6dV59Wa9495zGNSHC9ifQm0D0zVWDwQaB+LaTQIAgQIECBAgAABAsUKCLQXOzU61hYQaM9/Qcs3tLerzGMCBAgQIECgDwIC7d3MskB7N65aJUCAAAECfRIQaM9//m9YoD1CZ/Htqtdee+1AeUX47Ad+4AfqkFqEwW+++eZ6m/i22gjRNkNrEdiNQF51u+KKKwY+PBqhtwirNm8RbP6RH/mRup0IlDfD0BHGjFB4cz8RKo3AanWLgG98C2xzmwhzN0Oyb3rTm9LjHve4epsI4bUD4C996UvTscceW28T4cQIdFe3+D3CqM39/PM///PAh2Ej5PhTP/VT9TaPfvSj08aNG6smJu8jGNn8UpEIYN900031NhFAjuea+4mQY3yDbnWLgPUv/uIv1tvMnz8/Rci6eYvA66Me9ah6m1/+5V+eEl6NddbcT8x7/DdSdYswboREm9tEaLwZko3Q77Jly+ptfuInfmJKADza/f7v//56m9hvs54iMBpBz+Z+InDZDPFffvnlA/X0yEc+MsVzzVs4xfNVOxGib9ZTfItyBIer1+M+QqVRz9Xtk5/85EA9zZkzJ0W4vnmLYO3SpUvrdn7hF35hSj1FcPaYY46pt4nQe3zLcXWL8a9YsaJ+PfoS9dT8Zu7Yb7OeoobjAwXNW6yHRYsW1e3EB1QilF/d4gMKz3nOc+rXYz9//dd/PRDijxpt1tPChQvT+vXrqyYm7yOg/6M/+qN1O1EXzQ+oRE20w6oRNo65rW4RJI4PCDT9I0TbXKtvfetb0xOe8IR6m5/5mZ+ZUk8R0J03b169Tez3lltuqXYzGcCOD40093PhhRcOhK7jwwNxHKi2+bEf+7H0oQ99qG4jfokPP/zwD/9wvU0c85r1FMe7ODZWbcR9fBil+aGQj3/84wP1FP2ODwc0bxH8jw8XVe0sX758aD1Vr8f97//+70/+nx6qdmLOo96b28Rxr/nt6u9617vST/7kT9bbxLptB8DjG7cXLFhQbxPrpVlP8aGMCP839xP11Dz2R5tPfvKT620WL16cImTdvEUItn3sb4br41ve48Mczf3EB4qaHwqJ7Zv1NHv27BTHo+YtPmDQ/Abxn/u5n5vygYKop+ZajQ9RNIPm8SGGYfXU/Kb3WC9PetKT6v7Gl0XFBwqat/hwQDP8HUH/OHdS3eLDAhG+b445PowSz1e3+NBU89+7WKsxr81bBPTj356qnTj2N/+9i7Uax6Pq9biPD1HEv8dPfOITB55vbnMkv0cNxAcU2rXjG9qbMzeavwu0j+a86TUBAgQIECBAgACBUREQaB+VmdLPJNCe/4KWQLuFRYAAAQIECPRRQKC9m1kXaO/GVasECBAgQKBPAgLt+c//CbQLtDcDo3E8EWgXaI86EGiflQTaH3qHIdA+Kwm0f+//YHgkIfb238YHEiLE33xeoH3039kLtI/+HBoBAQIECBAgQIAAgZIFBNpLnh19GxAQaM9/QUugfaDEPCBAgAABAgR6IiDQ3s1EC7R346pVAgQIECDQJwGB9vzn/wTaBdoF2n1De4RJfUN7Sr6hfdbk/1nBN7Sn5Bvau/+G9vjG/ne/+93phS98oUD7mL2ZF2gfswk1HAIECBAgQIAAAQKFCQi0FzYhunNgAYH2/Be0BNoPXG9eIUCAAAECBMZXQKC9m7kVaO/GVasECBAgQKBPAgLt+c//DQu0P/3pT08XX3xx+upXvzpQXp/61KfS29/+9rRu3brJn2uuuSZ961vfqrf59re/nSIIWb0e95/73OfSvn376m3ivfb69evrbd7xjnekeK55++xnP5suvfTSepsPf/jDadeuXfUmDzzwQLr66qvr12M/Ecq+77776m2+/vWvpw9+8IMD29x+++3pwQcfrLfZsmVLes973lNvc9lll6UdO3bUr8cvH//4x9Nb3/rWepuPfexjaffu3fU28Xv0rznmL3zhC+m73/1uvc2XvvSlyf+7aLXNu971rnTnnXfWr8cv119/fbrkkkvqdv7zP/8z3XXXXfU23/nOd1I8V7UR9+G0d+/eepvt27enD3zgA/U20e877rijfj1++fznP5/+9V//td7mQx/6UAqr5i3mtbmfmPd777233uSb3/xmuvzyywe2ufXWW9P+/fvrbeLx+973vnqb97///Sn617xFu29729vqbT7ykY8M1FPs88orr6xfjz5F/++///66ma985SsD9fTOd74zxXPN22c+85kUz1djuuKKKwbqKeaqXU8xH1HP1W3nzp0D9fSWt7wlxbw2bzHvEc6s9hNzEX/XvEX9xN9W20R9Nesp1lP0r3o97qNOJyYm6maijuNaSLVN1HC7nmI9NNfqVVddNVBPMbb4BvaqjbiPeoo6q27RZrOeor0vf/nL1cuT97G+m/UUdfG1r32t3iZq4qMf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+xiUsSEQIECAAAECKSWB9vEsA4H28ZxXoyJAgAABAgQIECBQioBAeykzoR8HFbjhhhvSK17xivSiF73IT4cGZ599drrrrrsOOh82IECAAAECBAiMqkC813nd617nPWWH7ymr9+w33nhjmpiYGNVS0W8CBAgQIEDgKAsItAu0H+WSszsCBAgQIECgIwGB9o5gZ7hZgfYZngC7J0CAAAECBAgQIDDmAgLtYz7B4zS8e++9N91xxx3p9ttv99OhwdatW4WOxmnhGAsBAgQIECAwRSAC1nfeeaf3lB2+p6zes+/Zsyc9+OCDU+bAEwQIECBAgACBYQLtQPvxS5empz71lPTMZz7Lz2EanHbaM9Jxxx2Xmv+3x3PPPTft3Llz2BR4jgABAgQIECCQRaC0QPsP/dAPpZNPPjk9//nP93MEBj/90z+djj322Pq95amnnpquu+66LDWjEQIECBAgQIAAAQIECAi0qwECBAgQIECAAAECBAgQIECAAAECBAgQmHGBdqC9CmHPnj07+Tk8g8qweS/QPuOlrgMECBAgQGDsBUoLtFfvhbynPLz3lJVb5VjdC7SP/VI2QAIECBAgQIAAAQJHVUCg/ahy2xkBAgQIECBAgAABAgQIECBAgAABAgQIDBM4UKC9Csy4n1V/G+aRWAi0D6s+zxEgQIAAAQI5BUoNtB/Jeyh/O/W9qEB7zlWjLQIECBAgQIAAAQIEBNrVAAECBAgQIECAAAECBAgQIECAAAECBAjMuIBA+9SQUBfBKYH2GS91HSBAgAABAmMvINB+dN7XdfFe8VDaFGgf+6VsgAQIECBAgAABAgSOqoBA+1HltjMCBAgQIECAAAECBAgQIECAAAECBAgQGCZw6aWXprPPPju96lWv8tOhwRVXXJHuueeeYVPgOQIECBAgQIBAFoGZDrRfeeXG9JrXrkkvf/lZY/WzcuWq9PjHP37g/9qzfPnydNZZZ83I++eLL744bd26NUvNaIQAAQIECBAgQIAAAQIC7WqAAAECBAgQIECAAAECBAgQIECAAAECBGZc4K677krbtm3z07HB3XffnR544IEZn28dIECAAAECBMZXYKYD7du270i33HJb+sKWm8fq56qrr0mnnvq0gUB7fCD05ptvnpH30Lt27Ur79u0b30I2MgIECBAgQIAAAQIEjqqAQPtR5bYzAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECAwvgIzHWj/1t33pHH8uWnzlvTsZz9nINC+du3atHv37vEtJiMjQIAAAQIECBAgQKA3AgLtvZlqAyVAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAt0KCLR3E6gXaO+2brVOgAABAgQIECBAgMDMCgi0z6y/vRMgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgbEREGgXaB+bYjYQAgQIECBAgAABAgSOmoBA+1GjtiMCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIDDeAgLtAu3jXeFGR4AAAQIECBAgQIBAFwIC7V2oapMAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECPRQQKBdoL2HZW/IBAgQIECAAAECBAgcoYBA+xEC+nMCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIEDgIQGBdoF2a4EAAQIECBAgQIAAAQKHKiDQfqhitidAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgSGCgi0C7QPLQxPEiBAgAABAgQIECBA4GEEBNofBsdLBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAwPQFBNoF2qdfLbYkQIAAAQIECBAgQIDAQwIC7SqBAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQyCIg0C7QnqWQNEKAAAECBAgQIECAQK8EBNp7Nd0GS4AAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEuhMQaBdo7666tEyAAAECBAgQIECAwLgKCLSP68waFwECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQOMoCAu0C7Ue55OyOAAECBAgQIECAAIExEBBoH4NJNAQCBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIFCCgEC7QHsJdagPBAgQIECAAAECBAiMloBA+2jNl94SIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIFiBQTaBdqLLU4dI0CAAAECBAgQIECgWAGB9mKnRscIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMBoCQi0C7SPVsXqLQECBAgQIECAAAECJQgItJcwC/pAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgTGQECgXaB9DMrYEAgQIECAAAECBAgQOMoCAu1HGdzuCBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAuAoItAu0j2ttGxcBAgQIECBAgAABAt0JCLR3Z6tlAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECDQKwGBdoH2XhW8wRIgQIAAAQIECBAgkEVAoD0Lo0YIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABgXaBdquAAAECBAgQIECAAAEChyog0H6oYrYnQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEhgoItAu0Dy0MTxIgQIAAAQIECBAgQOBhBATaHwbHSwQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMD0BQTaBdqnXy22JECAAAECBAgQIECAwEMCAu0qgQABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIEMgiINAu0J6lkDRCgAABAgQIECBAgECvBATaezXdBkuAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBLoTEGgXaO+uurRMgAABAgQIECBAgMC4Cgi0j+vMGhcBAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIEDjKAgLtAu1HueTsjgABAgQIECBAgACBMRAQaB+DSTQEAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBQgoBAu0B7CXWoDwQIECBAgAABAgQIjJaAQPtozZfeEiBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgACBYgUE2gXaiy1OHSNAgAABAgQIECBAoFgBgfZip0bHCBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIDAaAkItAu0j1bF6i0BAgQIECBAgAABAiUICLSXMAv6QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIExkBgWKD9x3/8x9OLX/yStGbNuX4O02DlylXpMY99bJo1a1b9s3bt2rR79+4xqBpDIECAAAECBAgQIECg7wIC7X2vAOMnQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQKZBIYF2mfPnp0WLlyYHvnIJX4O02DBggVpzpw5dZg9gu0C7ZmKVjMECBAgQIAAAQIECMy4gED7jE+BDhAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAgfEQGBZob36ruN+/9w3rR2oh0D4ea8YoCBAgQIAAAQIECBBISaBdFRAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQJZBATa8wXWDxZ4F2jPUrIaIUCAAAECBAgQIECgAAGB9gImQRcIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgMA4CNx7773p/PPPT7/+67/up2ODSy+9NN13333jUDbGQIAAAQIECBAgQIBAzwUE2nteAIZPgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgRyCUxMTKStW7em//mf//HTscHOnTvTAw88kGvqtEOAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/BQTa+z3/Rk+AAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAIEZExBonzF6OyZAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgEC/Bf4fI0vgN2z0UuIAAAAASUVORK5CYII=)
  

    - UNet의 구조는 위 그림과 같다.
      - 많은 low-level information이 입력과 출력 과정에서 공유될 수 있다
      - encoder 네트웤의 출력을 그대로 decoder의 입력에도 가져다 쓰는 방식이다. 
      - 일종의 residual learning 방식이기에, encoder의 정보 외에 학습할 것들만 학습하면 되어, 학습 난이도가 낮아지고 더 좋은 성능을 보이게 된다. 


```python
from torch.nn.modules import padding
# U-Net 아키텍쳐의 다운 샘플링(Down Sampling) 모듈
class UNetDown(nn.Module):
  def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
    super(UNetDown, self).__init__()
    # 일반적인 conv layer의 형태이다. 채널은 깊어지고, 너비 높이는 감소하는!(downsampling)
    # 너비, 높이가 2배씩 감소
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
      layers.append(nn.InstanceNorm2d(out_channels))
    # LeakyReLU의 알파값이 0.2라는 소리, 입력이 음수일 때의 그래프의 기울기를 0.2로 설정한다는 의미. 
    layers.append(nn.LeakyReLU(0.2))
    if dropout:
      # dropout 은 해당 layer의 노드(weight)들 중 몇 퍼센트를 사용할 것인가를 결정
      layers.append(nn.Dropout(dropout))
    # layer 취합하여 model 하나로 만듬
    self.model = nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)

# U-Net 아키텍쳐의 업 샘플링 모듈: skip Connection 사용한다. 
# skip connection:  하나의 layer의 output을 몇 개의 layer를 건너뛰고 다음 layer의 input에 추가(concat)
class UNetUp(nn.Module):
  def __init__(self, in_channels, out_channels, dropout=0.0):
    super(UNetUp, self).__init__()
    # 일반적인 T conv layer의 형태이다. 채널은 얕아지고, 너비 높이는 증가하는!(upsampling)
    # 너비, 높이 2배씩 증가
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2 , padding=1, bias=False)]
    layers.append(nn.InstanceNorm2d(out_channels))
    # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
    layers.append(nn.ReLU(inplace=True))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)

  def forward(self, x, skip_input):
    x = self.model(x)
    # 위 UNetUp의 출력에 그대로 skip_input(encoder의 low-level info)을 더해줌
    x = torch.cat((x,skip_input),1) # 채널 레벨에서 합치기(concatenation), 채널이 두꺼워짐
    return x

# U-Net 생성자 아키텍쳐 
class GeneratorUNet(nn.Module):
  def __init__(self,in_channels=3,out_channels=3):
    super(GeneratorUNet, self).__init__()

    #encoder의 파트
    self.down1 = UNetDown(in_channels, 64, normalize=False) #출력: [64X128X128]
    self.down2 = UNetDown(64,128) #출력: [128X64X64]
    self.down3 = UNetDown(128,256) #출력: [256X32X32]
    self.down4 = UNetDown(256,512,dropout=0.5) #출력: [512X16X16]
    self.down5 = UNetDown(512,512,dropout=0.5) #출력: [512X8X8]
    self.down6 = UNetDown(512,512,dropout=0.5) #출력: [512X4X4]
    self.down7 = UNetDown(512,512,dropout=0.5) #출력: [512X2X2]
    self.down8 = UNetDown(512,512,normalize=False, dropout=0.5) #출력: [512X1X1], 그림상 중간 출력값

    #decoder의 파트
    #skip connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
    self.up1 = UNetUp(512,512,dropout=0.5) #출력: [1024X2X2]
    self.up2 = UNetUp(1024,512,dropout=0.5) #출력: [1024X4X4]
    self.up3 = UNetUp(1024,512,dropout=0.5) #출력: [1024X8X8]
    self.up4 = UNetUp(1024,512,dropout=0.5) #출력: [1024X16X16]
    self.up5 = UNetUp(1024,256) #출력: [512X32X32]
    self.up6 = UNetUp(512,128) #출력: [256X64X64]
    self.up7 = UNetUp(256,64) #출력: [128X128X128]

    self.final = nn.Sequential(
        # 출력 layer 크기 계산
        # OH = (H + 2P - FH) / S + 1
        # OW = (W + 2P - FW) / S + 1
        nn.Upsample(scale_factor=2), # 출력:[128X256X256]
        # pad_right, pad_left, pad_top, pad_bot
        nn.ZeroPad2d((1,0,1,0)),
        # 128 X 4 X 4 짜리 kernel이 3개 적용되서 output channel이 3이 나오게 될 것
        nn.Conv2d(128,out_channels,kernel_size=4,padding=1), #출력: [3X256X256]
        # 각 채널 별 256X256 이미지 pixel마다 Tanh()가 적용되어 -1과 1사이 출력을..!
        nn.Tanh(),
    )

  def forward(self,x):
    #인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    # 오른쪽 input은 skip_input으로 출력값에 그대로 채널 레벨로 더해진다(그림참고)
    u1 = self.up1(d8,d7)
    u2 = self.up2(u1,d6)
    u3 = self.up3(u2,d5)
    u4 = self.up4(u3,d4)
    u5 = self.up5(u4,d3)
    u6 = self.up6(u5,d2)
    u7 = self.up7(u6,d1)

    return self.final(u7)
  

class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super(Discriminator,self).__init__()

    def discriminator_block(in_channels, out_channels, normalization=True):
      #너비와 높이가 2배씩 감소
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
      if normalization:
        layers.append(nn.InstanceNorm2d(out_channels))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers
      
    self.model = nn.Sequential(
        #두 개의 이미지(실제/변환된 이미지, 조건 이미지)를 한꺼번에 입력 받으므로 입력 채널의 크기는 2배(6)가 된다
        *discriminator_block(in_channels *2, 64, normalization=False),# 출력: [64 X 128 X 128]
        *discriminator_block(64, 128), # 출력: [128 X 64 X 64]
        *discriminator_block(128,256), # 출력: [256 X 32 X 32]
        *discriminator_block(256,512), # 출력: [512 X 16 X 16]
        nn.ZeroPad2d((1,0,1,0)),
        nn.Conv2d(512,1,kernel_size=4,padding=1,bias=False) #출력[1X16X16]
    )

    #img_A: 실제/변환된 이미지, img_B: 조건(condition)
  def forward(self, img_A, img_B):
    #이미지 두개를 채널 레벨에서 연결하여(concatenate) 입력 데이터 생성
    img_input = torch.cat((img_A,img_B),1)
    return self.model(img_input)
    
    
```

## 모델 학습 및 샘플링
 - 학습을 위해 G와 D 모델 초기화
 - 적절한 하이퍼파라미터 설정
 - 적절한 손실 함수 사용
  - pix2pix는 L1 loss를 이용해서 출력 이미지가 ground-truth와 유사해질 수 있도록 노력(blurry함 감소)


```python
def weights_init_normal(m):
  # __name__ : 사용되는 현재 모듈의 이름을 제공하는 특수 내장 변수
  classname = m.__class__.__name__
  # find()는 찾는 해당 문자열이 없으면 -1을 반환
  # "Conv"라면 
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  # "BatchNorm2d"라면
  elif classname.find("BatchNorm2d") != -1:
    # 정규분포에서 sampling한 값으로 초기화 (mean: 1.0 / std = 0.02의 정규뷴포에서)
    # 주어진 2차원의 tensor(=m.weight.data)를 정규분포에서 뽑은 값으로 초기화한다. 
    # 이 작업에서 gradient가 기록되지는 않는다. 
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    # 명시한 scalr 값으로 tensor를 inplace로 채움  
    torch.nn.init.constant_(m.bias.data, 0.0)

# 생성자와 판별자 초기화
generator = GeneratorUNet()
discriminator = Discriminator()

# This package adds support for CUDA tensor types, 
# that implement the same function as CPU tensors, 
# ,but they utilize GPUs for computation. 
generator.cuda()
discriminator.cuda()

# 가중치 초기화 / apply(fn)
# Applies fn recursively to every submodule (as returned by .children()) as well as self. 
# Typical use includes initializing the parameters of a model
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 손실 함수(loss function)
# cGAN을 위해 MSE loss를 사용한다 
# 실제 정답 이미지와 비슷해지기 위해 pixel wise로 L1 loss 사용 
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

criterion_GAN.cuda()
criterion_pixelwise.cuda()

# 학습률 설정
lr = 0.0002

# 생성자와 판별자를 위한 최적화 함수
# parameters(): Returns an iterator over module parameters. This is typically passed to an optimizer.
# betas : coefficients used for computing running averages of gradient and its square 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))


```

 - 모델을 학습하면서 주기적으로 샘플링하여 결과를 확인할 수 있다. 


```python
import time

n_epochs = 200 # 학습의 횟수(epoch) 설정
sample_interval = 200 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정 

# 변환된 이미지와 정답 이미지 사이의 L1 픽셀 단위(Pixel-wise) 손실 가중치(weight) 파라미터 
lambda_pixel = 100

start_time = time.time()

for epoch in range(n_epochs):

  # enumerate는 튜플로 iterate한다. 
  # unpacking 시켜서 보통 loop을 돌린다(아래 i, batch 처럼)
  # i는 인덱스로 0,1,2 ,,, 로 
  # batch는 해당 train_dataloader 리스트의 iterable 객체를(보통 string) 함께 iterate한다 
  # ex) (0, "A") , (1, "B")
  for i, batch in enumerate(train_dataloader):
    ########################################################
    # 모델의 입력(input) 데이터 불러오기
    # train_dataloader는 ImageDataset을 input으로 불러온 데이터셋 
    #: {"A": img_A, "B": img_B} 형태를 return한다. 
    # img_A: 실제 / 변환생성된 이미지
    # img_B: 조건 이미지
    # 정답 조건이미지
    real_A = batch["B"].cuda()
    # 정답 변환이미지
    real_B = batch["A"].cuda()
    ########################################################

    real = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0) # 진짜(real): 1
    fake = torch.cuda.FloatTensor(real_B.size(0), 1, 16, 16).fill_(0.0) # 가짜(fake): 0

    """ 생성자를 학습시킨다 """
    # Sets the gradients of all optimized torch.Tensor s to zero.
    optimizer_G.zero_grad()

    # 주어진 조건 real_A 이미지를 갖고 변환 생성한 가짜 이미지
    fake_B = generator(real_A)

    # 생성자의 손실 계산
    # criterion_GAN은 torch.nn.MSELoss()이다 즉, label 'real'과의 Loss를 계산 
    # fake_B와 real_A를 channel 레벨로 concat
    loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)

    # 픽셀 단위 L1 손실 값 계산
    # torch.nn.L1Loss()이다. 'real_B'즉, 정답 변환이미지와 생성한 변환이미지 사이 L1 loss 계산
    loss_pixel = criterion_pixelwise(fake_B, real_B)

    ########################################################
    # 최종 loss
    loss_G = loss_GAN + lambda_pixel * loss_pixel
    ########################################################

    ########################################################
    # 생성자 업데이트
    # pytorch의 automatic gradient package인 autograd의 함수를 쓰는 것 
    # 즉, 이 loss(scalar)값을 모든 가중치에 대해 미분한다. 
    # backward()는 gradient를 구하려고 하는 변수들에 대해 Dynamic computational graph에 차곡차곡 neural net의 계산 흐름을 쌓아 둔다 
    loss_G.backward()
    # step(): Performs a single optimization step.
    optimizer_G.step()
    ########################################################

    """ 판별자를 학습시킨다 """
    optimizer_D.zero_grad()

    # 판별자의 손실 값 계산 
    loss_real = criterion_GAN(discriminator(real_B, real_A), real) #조건: real_A 
    loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
    loss_D = (loss_real + loss_fake) /2

    # 판별자 업데이트 
    loss_D.backward()
    optimizer_D.step()

    done = epoch*len(train_dataloader) + i
    if done % sample_interval == 0:
      imgs = next(iter(val_dataloader)) # 10개 이미지 추출하여 생성
      real_A = imgs["B"].cuda()
      real_B = imgs["A"].cuda()
      fake_B = generator(real_A)
      # real_A : 조건(condition), fake_B: 변환된 이미지, real_B: 정답 변환 이미지
      img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # 높이(height)를 기준으로 
      save_image(img_sample, f"{done}.png", nrow=5, normalize=True)

  print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {time.time() - start_time:.2f}s]")
  

```

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))


    [Epoch 0/200] [D loss: 0.381288] [G pixel loss: 0.376942, adv loss: 0.6545624732971191] [Elapsed time: 49.09s]
    [Epoch 1/200] [D loss: 0.192580] [G pixel loss: 0.355091, adv loss: 0.7536435127258301] [Elapsed time: 97.05s]
    [Epoch 2/200] [D loss: 0.149096] [G pixel loss: 0.345436, adv loss: 0.7810912132263184] [Elapsed time: 145.01s]
    [Epoch 3/200] [D loss: 0.225314] [G pixel loss: 0.341125, adv loss: 1.3068674802780151] [Elapsed time: 193.78s]
    [Epoch 4/200] [D loss: 0.143116] [G pixel loss: 0.353989, adv loss: 0.5862461924552917] [Elapsed time: 241.68s]
    [Epoch 5/200] [D loss: 0.054412] [G pixel loss: 0.331029, adv loss: 0.9602774381637573] [Elapsed time: 289.60s]
    [Epoch 6/200] [D loss: 0.052271] [G pixel loss: 0.367796, adv loss: 0.7469775676727295] [Elapsed time: 337.40s]
    [Epoch 7/200] [D loss: 0.174204] [G pixel loss: 0.325465, adv loss: 0.471322238445282] [Elapsed time: 386.14s]
    [Epoch 8/200] [D loss: 0.078990] [G pixel loss: 0.341791, adv loss: 0.5842515230178833] [Elapsed time: 434.01s]
    [Epoch 9/200] [D loss: 0.063996] [G pixel loss: 0.342018, adv loss: 0.7872447967529297] [Elapsed time: 481.80s]
    [Epoch 10/200] [D loss: 0.066601] [G pixel loss: 0.362324, adv loss: 0.6874423027038574] [Elapsed time: 529.70s]
    [Epoch 11/200] [D loss: 0.076826] [G pixel loss: 0.289537, adv loss: 0.6844277381896973] [Elapsed time: 578.38s]
    [Epoch 12/200] [D loss: 0.085551] [G pixel loss: 0.309521, adv loss: 0.9681031703948975] [Elapsed time: 626.23s]
    [Epoch 13/200] [D loss: 0.039946] [G pixel loss: 0.394171, adv loss: 1.1878159046173096] [Elapsed time: 674.26s]
    [Epoch 14/200] [D loss: 0.057577] [G pixel loss: 0.310740, adv loss: 0.9615353941917419] [Elapsed time: 722.18s]
    [Epoch 15/200] [D loss: 0.050391] [G pixel loss: 0.368409, adv loss: 0.8237900733947754] [Elapsed time: 771.12s]
    [Epoch 16/200] [D loss: 0.105165] [G pixel loss: 0.379902, adv loss: 0.6350650787353516] [Elapsed time: 819.03s]
    [Epoch 17/200] [D loss: 0.041258] [G pixel loss: 0.309569, adv loss: 0.9446393847465515] [Elapsed time: 867.05s]
    [Epoch 18/200] [D loss: 0.088850] [G pixel loss: 0.256369, adv loss: 1.3284671306610107] [Elapsed time: 914.94s]
    [Epoch 19/200] [D loss: 0.065282] [G pixel loss: 0.315536, adv loss: 0.6887872219085693] [Elapsed time: 963.83s]
    [Epoch 20/200] [D loss: 0.042505] [G pixel loss: 0.289952, adv loss: 0.8647300004959106] [Elapsed time: 1011.81s]
    [Epoch 21/200] [D loss: 0.060231] [G pixel loss: 0.313320, adv loss: 0.7799541354179382] [Elapsed time: 1059.78s]
    [Epoch 22/200] [D loss: 0.039047] [G pixel loss: 0.308499, adv loss: 0.7362779378890991] [Elapsed time: 1107.59s]
    [Epoch 23/200] [D loss: 0.079212] [G pixel loss: 0.297559, adv loss: 0.6167521476745605] [Elapsed time: 1156.35s]
    [Epoch 24/200] [D loss: 0.110600] [G pixel loss: 0.297773, adv loss: 1.5952587127685547] [Elapsed time: 1204.19s]
    [Epoch 25/200] [D loss: 0.065958] [G pixel loss: 0.315930, adv loss: 0.5173804759979248] [Elapsed time: 1251.93s]
    [Epoch 26/200] [D loss: 0.035456] [G pixel loss: 0.289858, adv loss: 1.190037727355957] [Elapsed time: 1299.71s]
    [Epoch 27/200] [D loss: 0.051859] [G pixel loss: 0.293352, adv loss: 1.2759935855865479] [Elapsed time: 1348.53s]
    [Epoch 28/200] [D loss: 0.067366] [G pixel loss: 0.299187, adv loss: 0.8222799301147461] [Elapsed time: 1396.56s]
    [Epoch 29/200] [D loss: 0.068348] [G pixel loss: 0.255664, adv loss: 1.340897560119629] [Elapsed time: 1444.37s]
    [Epoch 30/200] [D loss: 0.089547] [G pixel loss: 0.291004, adv loss: 0.5226625800132751] [Elapsed time: 1492.13s]
    [Epoch 31/200] [D loss: 0.081532] [G pixel loss: 0.269906, adv loss: 0.7131866812705994] [Elapsed time: 1540.93s]
    [Epoch 32/200] [D loss: 0.061981] [G pixel loss: 0.259353, adv loss: 0.6018949151039124] [Elapsed time: 1588.93s]
    [Epoch 33/200] [D loss: 0.060490] [G pixel loss: 0.265419, adv loss: 0.6979140043258667] [Elapsed time: 1636.79s]
    [Epoch 34/200] [D loss: 0.050280] [G pixel loss: 0.263957, adv loss: 1.1079914569854736] [Elapsed time: 1684.86s]
    [Epoch 35/200] [D loss: 0.065202] [G pixel loss: 0.260654, adv loss: 0.5337224006652832] [Elapsed time: 1733.74s]
    [Epoch 36/200] [D loss: 0.069398] [G pixel loss: 0.285417, adv loss: 0.5545682311058044] [Elapsed time: 1781.70s]
    [Epoch 37/200] [D loss: 0.027014] [G pixel loss: 0.295575, adv loss: 0.8494169116020203] [Elapsed time: 1829.59s]
    [Epoch 38/200] [D loss: 0.074418] [G pixel loss: 0.275254, adv loss: 0.5166867971420288] [Elapsed time: 1877.59s]
    [Epoch 39/200] [D loss: 0.078900] [G pixel loss: 0.229593, adv loss: 1.2745400667190552] [Elapsed time: 1926.48s]
    [Epoch 40/200] [D loss: 0.049240] [G pixel loss: 0.257130, adv loss: 0.7442737817764282] [Elapsed time: 1974.52s]
    [Epoch 41/200] [D loss: 0.118156] [G pixel loss: 0.279518, adv loss: 0.4168187975883484] [Elapsed time: 2022.49s]
    [Epoch 42/200] [D loss: 0.045100] [G pixel loss: 0.264872, adv loss: 0.6888045072555542] [Elapsed time: 2070.29s]
    [Epoch 43/200] [D loss: 0.042551] [G pixel loss: 0.258340, adv loss: 0.7977311015129089] [Elapsed time: 2119.30s]
    [Epoch 44/200] [D loss: 0.025458] [G pixel loss: 0.248376, adv loss: 1.0759673118591309] [Elapsed time: 2167.28s]
    [Epoch 45/200] [D loss: 0.110448] [G pixel loss: 0.256910, adv loss: 0.38716554641723633] [Elapsed time: 2215.10s]
    [Epoch 46/200] [D loss: 0.087167] [G pixel loss: 0.261619, adv loss: 1.1514577865600586] [Elapsed time: 2262.96s]
    [Epoch 47/200] [D loss: 0.040348] [G pixel loss: 0.257065, adv loss: 1.0923411846160889] [Elapsed time: 2311.64s]
    [Epoch 48/200] [D loss: 0.077049] [G pixel loss: 0.255525, adv loss: 1.0998553037643433] [Elapsed time: 2359.48s]
    [Epoch 49/200] [D loss: 0.044824] [G pixel loss: 0.218836, adv loss: 1.2029449939727783] [Elapsed time: 2407.46s]
    [Epoch 50/200] [D loss: 0.037130] [G pixel loss: 0.264936, adv loss: 0.93736332654953] [Elapsed time: 2456.60s]
    [Epoch 51/200] [D loss: 0.035223] [G pixel loss: 0.235452, adv loss: 0.8149962425231934] [Elapsed time: 2504.60s]
    [Epoch 52/200] [D loss: 0.028359] [G pixel loss: 0.212014, adv loss: 0.9162178039550781] [Elapsed time: 2552.50s]
    [Epoch 53/200] [D loss: 0.115831] [G pixel loss: 0.228224, adv loss: 1.1170363426208496] [Elapsed time: 2600.35s]
    [Epoch 54/200] [D loss: 0.029847] [G pixel loss: 0.233618, adv loss: 0.9819955825805664] [Elapsed time: 2649.14s]
    [Epoch 55/200] [D loss: 0.041420] [G pixel loss: 0.227949, adv loss: 0.6626397371292114] [Elapsed time: 2697.16s]
    [Epoch 56/200] [D loss: 0.043162] [G pixel loss: 0.245179, adv loss: 0.9243547320365906] [Elapsed time: 2745.11s]
    [Epoch 57/200] [D loss: 0.042623] [G pixel loss: 0.225400, adv loss: 0.7696890830993652] [Elapsed time: 2793.19s]
    [Epoch 58/200] [D loss: 0.038320] [G pixel loss: 0.239521, adv loss: 1.221077799797058] [Elapsed time: 2841.93s]
    [Epoch 59/200] [D loss: 0.041262] [G pixel loss: 0.198182, adv loss: 0.8226012587547302] [Elapsed time: 2889.78s]
    [Epoch 60/200] [D loss: 0.105454] [G pixel loss: 0.226028, adv loss: 0.4266689717769623] [Elapsed time: 2937.50s]
    [Epoch 61/200] [D loss: 0.059939] [G pixel loss: 0.227845, adv loss: 0.7188699245452881] [Elapsed time: 2985.56s]
    [Epoch 62/200] [D loss: 0.024269] [G pixel loss: 0.212256, adv loss: 0.9303908348083496] [Elapsed time: 3034.26s]
    [Epoch 63/200] [D loss: 0.066482] [G pixel loss: 0.217997, adv loss: 0.839183509349823] [Elapsed time: 3082.26s]
    [Epoch 64/200] [D loss: 0.080208] [G pixel loss: 0.228155, adv loss: 0.943450391292572] [Elapsed time: 3130.19s]
    [Epoch 65/200] [D loss: 0.069581] [G pixel loss: 0.226142, adv loss: 0.5254068374633789] [Elapsed time: 3178.20s]
    [Epoch 66/200] [D loss: 0.050510] [G pixel loss: 0.218243, adv loss: 1.1991784572601318] [Elapsed time: 3227.02s]
    [Epoch 67/200] [D loss: 0.102993] [G pixel loss: 0.235961, adv loss: 1.3224527835845947] [Elapsed time: 3274.98s]
    [Epoch 68/200] [D loss: 0.052456] [G pixel loss: 0.232487, adv loss: 0.9387426376342773] [Elapsed time: 3322.92s]
    [Epoch 69/200] [D loss: 0.056939] [G pixel loss: 0.237723, adv loss: 0.5924131274223328] [Elapsed time: 3370.88s]
    [Epoch 70/200] [D loss: 0.051492] [G pixel loss: 0.200669, adv loss: 0.883420467376709] [Elapsed time: 3419.79s]
    [Epoch 71/200] [D loss: 0.092826] [G pixel loss: 0.228369, adv loss: 0.5278297662734985] [Elapsed time: 3467.64s]
    [Epoch 72/200] [D loss: 0.038645] [G pixel loss: 0.182996, adv loss: 0.7274574041366577] [Elapsed time: 3515.41s]
    [Epoch 73/200] [D loss: 0.035151] [G pixel loss: 0.239637, adv loss: 0.7851053476333618] [Elapsed time: 3563.28s]
    [Epoch 74/200] [D loss: 0.055824] [G pixel loss: 0.205222, adv loss: 0.5974189043045044] [Elapsed time: 3612.04s]
    [Epoch 75/200] [D loss: 0.029098] [G pixel loss: 0.216047, adv loss: 1.0696113109588623] [Elapsed time: 3659.88s]
    [Epoch 76/200] [D loss: 0.041786] [G pixel loss: 0.205262, adv loss: 1.0595576763153076] [Elapsed time: 3707.85s]
    [Epoch 77/200] [D loss: 0.054658] [G pixel loss: 0.200384, adv loss: 0.9168607592582703] [Elapsed time: 3755.88s]
    [Epoch 78/200] [D loss: 0.037715] [G pixel loss: 0.221479, adv loss: 1.2193284034729004] [Elapsed time: 3804.59s]
    [Epoch 79/200] [D loss: 0.032999] [G pixel loss: 0.225056, adv loss: 0.7432982325553894] [Elapsed time: 3852.51s]
    [Epoch 80/200] [D loss: 0.028037] [G pixel loss: 0.233739, adv loss: 0.9614901542663574] [Elapsed time: 3900.45s]
    [Epoch 81/200] [D loss: 0.066227] [G pixel loss: 0.211871, adv loss: 0.5838621854782104] [Elapsed time: 3948.24s]
    [Epoch 82/200] [D loss: 0.086552] [G pixel loss: 0.222979, adv loss: 0.5007441639900208] [Elapsed time: 3997.05s]
    [Epoch 83/200] [D loss: 0.020661] [G pixel loss: 0.191860, adv loss: 0.9374706149101257] [Elapsed time: 4045.01s]
    [Epoch 84/200] [D loss: 0.029254] [G pixel loss: 0.223549, adv loss: 0.9884688854217529] [Elapsed time: 4092.92s]
    [Epoch 85/200] [D loss: 0.063913] [G pixel loss: 0.188804, adv loss: 0.9794902801513672] [Elapsed time: 4140.78s]
    [Epoch 86/200] [D loss: 0.032603] [G pixel loss: 0.207368, adv loss: 1.1193656921386719] [Elapsed time: 4189.52s]
    [Epoch 87/200] [D loss: 0.050683] [G pixel loss: 0.198689, adv loss: 0.6600277423858643] [Elapsed time: 4237.52s]
    [Epoch 88/200] [D loss: 0.052047] [G pixel loss: 0.184886, adv loss: 1.0955743789672852] [Elapsed time: 4285.33s]
    [Epoch 89/200] [D loss: 0.032133] [G pixel loss: 0.213616, adv loss: 1.2789306640625] [Elapsed time: 4333.42s]
    [Epoch 90/200] [D loss: 0.055240] [G pixel loss: 0.195986, adv loss: 0.7520836591720581] [Elapsed time: 4382.29s]
    [Epoch 91/200] [D loss: 0.018873] [G pixel loss: 0.250362, adv loss: 0.9447711706161499] [Elapsed time: 4430.28s]
    [Epoch 92/200] [D loss: 0.034829] [G pixel loss: 0.191047, adv loss: 1.0977423191070557] [Elapsed time: 4478.27s]
    [Epoch 93/200] [D loss: 0.036279] [G pixel loss: 0.197346, adv loss: 0.9610500335693359] [Elapsed time: 4526.41s]
    [Epoch 94/200] [D loss: 0.054154] [G pixel loss: 0.193912, adv loss: 1.1074565649032593] [Elapsed time: 4575.23s]
    [Epoch 95/200] [D loss: 0.025659] [G pixel loss: 0.217339, adv loss: 0.8696063756942749] [Elapsed time: 4623.13s]
    [Epoch 96/200] [D loss: 0.024971] [G pixel loss: 0.227698, adv loss: 1.3020386695861816] [Elapsed time: 4671.02s]
    [Epoch 97/200] [D loss: 0.051727] [G pixel loss: 0.190971, adv loss: 1.073716402053833] [Elapsed time: 4719.07s]
    [Epoch 98/200] [D loss: 0.092976] [G pixel loss: 0.185161, adv loss: 1.2253074645996094] [Elapsed time: 4767.86s]
    [Epoch 99/200] [D loss: 0.030444] [G pixel loss: 0.194988, adv loss: 0.9471611976623535] [Elapsed time: 4815.69s]
    [Epoch 100/200] [D loss: 0.022689] [G pixel loss: 0.182422, adv loss: 1.0049575567245483] [Elapsed time: 4863.62s]
    [Epoch 101/200] [D loss: 0.033428] [G pixel loss: 0.220642, adv loss: 0.7620983123779297] [Elapsed time: 4912.43s]
    [Epoch 102/200] [D loss: 0.027239] [G pixel loss: 0.190845, adv loss: 0.9133728742599487] [Elapsed time: 4960.32s]
    [Epoch 103/200] [D loss: 0.054579] [G pixel loss: 0.192445, adv loss: 0.9957443475723267] [Elapsed time: 5008.10s]
    [Epoch 104/200] [D loss: 0.031166] [G pixel loss: 0.195483, adv loss: 1.0524855852127075] [Elapsed time: 5056.14s]
    [Epoch 105/200] [D loss: 0.134346] [G pixel loss: 0.185030, adv loss: 0.35340946912765503] [Elapsed time: 5104.96s]
    [Epoch 106/200] [D loss: 0.036587] [G pixel loss: 0.194746, adv loss: 0.793824315071106] [Elapsed time: 5152.74s]
    [Epoch 107/200] [D loss: 0.038521] [G pixel loss: 0.196390, adv loss: 1.0377753973007202] [Elapsed time: 5200.65s]
    [Epoch 108/200] [D loss: 0.053480] [G pixel loss: 0.198559, adv loss: 0.6999761462211609] [Elapsed time: 5248.49s]
    [Epoch 109/200] [D loss: 0.045676] [G pixel loss: 0.187569, adv loss: 0.6513010859489441] [Elapsed time: 5297.36s]
    [Epoch 110/200] [D loss: 0.069132] [G pixel loss: 0.183542, adv loss: 0.6066778898239136] [Elapsed time: 5345.24s]
    [Epoch 111/200] [D loss: 0.026282] [G pixel loss: 0.172196, adv loss: 0.8419086337089539] [Elapsed time: 5393.18s]
    [Epoch 112/200] [D loss: 0.066565] [G pixel loss: 0.194839, adv loss: 0.898115336894989] [Elapsed time: 5441.21s]
    [Epoch 113/200] [D loss: 0.018404] [G pixel loss: 0.195488, adv loss: 1.0069154500961304] [Elapsed time: 5490.05s]
    [Epoch 114/200] [D loss: 0.034999] [G pixel loss: 0.174557, adv loss: 0.9553066492080688] [Elapsed time: 5538.05s]
    [Epoch 115/200] [D loss: 0.028621] [G pixel loss: 0.201182, adv loss: 1.1179637908935547] [Elapsed time: 5585.99s]
    [Epoch 116/200] [D loss: 0.033847] [G pixel loss: 0.207996, adv loss: 1.0239548683166504] [Elapsed time: 5633.97s]
    [Epoch 117/200] [D loss: 3.286136] [G pixel loss: 0.196184, adv loss: 3.55122447013855] [Elapsed time: 5682.81s]
    [Epoch 118/200] [D loss: 0.074369] [G pixel loss: 0.181119, adv loss: 0.7259426116943359] [Elapsed time: 5730.55s]
    [Epoch 119/200] [D loss: 0.088055] [G pixel loss: 0.159730, adv loss: 1.0089730024337769] [Elapsed time: 5778.34s]
    [Epoch 120/200] [D loss: 0.081245] [G pixel loss: 0.180452, adv loss: 1.197155237197876] [Elapsed time: 5826.11s]
    [Epoch 121/200] [D loss: 0.026494] [G pixel loss: 0.188146, adv loss: 0.9939560890197754] [Elapsed time: 5874.89s]
    [Epoch 122/200] [D loss: 0.040258] [G pixel loss: 0.164458, adv loss: 0.8700642585754395] [Elapsed time: 5922.78s]
    [Epoch 123/200] [D loss: 0.024400] [G pixel loss: 0.190870, adv loss: 1.0706164836883545] [Elapsed time: 5970.63s]
    [Epoch 124/200] [D loss: 0.029007] [G pixel loss: 0.192408, adv loss: 0.7650015354156494] [Elapsed time: 6018.56s]
    [Epoch 125/200] [D loss: 0.045748] [G pixel loss: 0.163438, adv loss: 0.6413494348526001] [Elapsed time: 6067.22s]
    [Epoch 126/200] [D loss: 0.041828] [G pixel loss: 0.177025, adv loss: 1.3107445240020752] [Elapsed time: 6115.22s]
    [Epoch 127/200] [D loss: 0.055553] [G pixel loss: 0.180761, adv loss: 1.100488543510437] [Elapsed time: 6163.18s]
    [Epoch 128/200] [D loss: 0.043807] [G pixel loss: 0.197879, adv loss: 1.1993601322174072] [Elapsed time: 6211.23s]
    [Epoch 129/200] [D loss: 0.021107] [G pixel loss: 0.178910, adv loss: 1.0723681449890137] [Elapsed time: 6260.17s]
    [Epoch 130/200] [D loss: 0.049353] [G pixel loss: 0.171260, adv loss: 1.2647234201431274] [Elapsed time: 6308.19s]
    [Epoch 131/200] [D loss: 0.035712] [G pixel loss: 0.199795, adv loss: 0.9335449934005737] [Elapsed time: 6356.08s]
    [Epoch 132/200] [D loss: 0.039342] [G pixel loss: 0.194386, adv loss: 0.8128072023391724] [Elapsed time: 6404.05s]
    [Epoch 133/200] [D loss: 0.024539] [G pixel loss: 0.175065, adv loss: 0.7909610271453857] [Elapsed time: 6452.74s]
    [Epoch 134/200] [D loss: 0.024563] [G pixel loss: 0.193471, adv loss: 1.0189177989959717] [Elapsed time: 6500.67s]
    [Epoch 135/200] [D loss: 0.029262] [G pixel loss: 0.178664, adv loss: 0.7336101531982422] [Elapsed time: 6548.55s]
    [Epoch 136/200] [D loss: 0.040150] [G pixel loss: 0.188662, adv loss: 0.6959328651428223] [Elapsed time: 6596.64s]
    [Epoch 137/200] [D loss: 0.048452] [G pixel loss: 0.189568, adv loss: 0.6468292474746704] [Elapsed time: 6645.47s]
    [Epoch 138/200] [D loss: 0.014524] [G pixel loss: 0.181803, adv loss: 0.9406778216362] [Elapsed time: 6693.48s]
    [Epoch 139/200] [D loss: 0.016484] [G pixel loss: 0.186653, adv loss: 0.883161723613739] [Elapsed time: 6741.37s]
    [Epoch 140/200] [D loss: 0.035722] [G pixel loss: 0.205902, adv loss: 0.7527605891227722] [Elapsed time: 6789.29s]
    [Epoch 141/200] [D loss: 0.021733] [G pixel loss: 0.162444, adv loss: 0.8068457841873169] [Elapsed time: 6838.00s]
    [Epoch 142/200] [D loss: 0.062122] [G pixel loss: 0.171460, adv loss: 1.2483985424041748] [Elapsed time: 6885.86s]
    [Epoch 143/200] [D loss: 0.040015] [G pixel loss: 0.203298, adv loss: 1.3327226638793945] [Elapsed time: 6933.87s]
    [Epoch 144/200] [D loss: 0.041189] [G pixel loss: 0.181297, adv loss: 0.8710848093032837] [Elapsed time: 6981.81s]
    [Epoch 145/200] [D loss: 0.026046] [G pixel loss: 0.184190, adv loss: 0.8048621416091919] [Elapsed time: 7030.70s]
    [Epoch 146/200] [D loss: 0.020195] [G pixel loss: 0.183135, adv loss: 0.9086407423019409] [Elapsed time: 7078.59s]
    [Epoch 147/200] [D loss: 0.035239] [G pixel loss: 0.161495, adv loss: 1.1328773498535156] [Elapsed time: 7126.54s]
    [Epoch 148/200] [D loss: 0.027274] [G pixel loss: 0.187972, adv loss: 0.8399996757507324] [Elapsed time: 7174.47s]
    [Epoch 149/200] [D loss: 0.019699] [G pixel loss: 0.164747, adv loss: 1.0377000570297241] [Elapsed time: 7223.46s]
    [Epoch 150/200] [D loss: 0.015246] [G pixel loss: 0.195141, adv loss: 1.0458558797836304] [Elapsed time: 7271.23s]
    [Epoch 151/200] [D loss: 0.019110] [G pixel loss: 0.197082, adv loss: 0.9519748687744141] [Elapsed time: 7318.95s]
    [Epoch 152/200] [D loss: 0.056855] [G pixel loss: 0.182716, adv loss: 0.6151430606842041] [Elapsed time: 7367.82s]
    [Epoch 153/200] [D loss: 0.033678] [G pixel loss: 0.177233, adv loss: 0.9871024489402771] [Elapsed time: 7415.73s]
    [Epoch 154/200] [D loss: 0.031658] [G pixel loss: 0.170103, adv loss: 1.1817864179611206] [Elapsed time: 7463.68s]
    [Epoch 155/200] [D loss: 0.028353] [G pixel loss: 0.177376, adv loss: 0.7463202476501465] [Elapsed time: 7511.53s]
    [Epoch 156/200] [D loss: 0.025022] [G pixel loss: 0.179925, adv loss: 0.9695056676864624] [Elapsed time: 7560.43s]
    [Epoch 157/200] [D loss: 0.032364] [G pixel loss: 0.146468, adv loss: 0.789677619934082] [Elapsed time: 7608.50s]
    [Epoch 158/200] [D loss: 0.028525] [G pixel loss: 0.179535, adv loss: 0.7461600303649902] [Elapsed time: 7656.57s]
    [Epoch 159/200] [D loss: 0.028537] [G pixel loss: 0.180690, adv loss: 1.1370570659637451] [Elapsed time: 7704.46s]
    [Epoch 160/200] [D loss: 0.014208] [G pixel loss: 0.171034, adv loss: 1.1104817390441895] [Elapsed time: 7753.40s]
    [Epoch 161/200] [D loss: 0.036036] [G pixel loss: 0.180282, adv loss: 0.668064296245575] [Elapsed time: 7801.32s]
    [Epoch 162/200] [D loss: 0.017487] [G pixel loss: 0.151971, adv loss: 0.9679929614067078] [Elapsed time: 7849.27s]
    [Epoch 163/200] [D loss: 0.017655] [G pixel loss: 0.164969, adv loss: 0.8295966982841492] [Elapsed time: 7897.32s]
    [Epoch 164/200] [D loss: 0.015272] [G pixel loss: 0.177389, adv loss: 0.9722349047660828] [Elapsed time: 7946.07s]
    [Epoch 165/200] [D loss: 0.027824] [G pixel loss: 0.170566, adv loss: 1.2339824438095093] [Elapsed time: 7994.11s]
    [Epoch 166/200] [D loss: 0.027167] [G pixel loss: 0.174402, adv loss: 0.7651223540306091] [Elapsed time: 8042.09s]
    [Epoch 167/200] [D loss: 0.016944] [G pixel loss: 0.175642, adv loss: 0.8624167442321777] [Elapsed time: 8089.94s]
    [Epoch 168/200] [D loss: 0.011771] [G pixel loss: 0.175168, adv loss: 1.0963164567947388] [Elapsed time: 8138.83s]
    [Epoch 169/200] [D loss: 0.015328] [G pixel loss: 0.178143, adv loss: 0.9898573756217957] [Elapsed time: 8186.68s]
    [Epoch 170/200] [D loss: 0.040983] [G pixel loss: 0.192682, adv loss: 0.674818754196167] [Elapsed time: 8234.59s]
    [Epoch 171/200] [D loss: 0.015219] [G pixel loss: 0.185358, adv loss: 1.1193597316741943] [Elapsed time: 8282.56s]
    [Epoch 172/200] [D loss: 0.018121] [G pixel loss: 0.174457, adv loss: 0.8547428846359253] [Elapsed time: 8331.55s]
    [Epoch 173/200] [D loss: 0.030641] [G pixel loss: 0.186524, adv loss: 0.8115360736846924] [Elapsed time: 8379.49s]
    [Epoch 174/200] [D loss: 0.016824] [G pixel loss: 0.171252, adv loss: 0.894091784954071] [Elapsed time: 8427.49s]
    [Epoch 175/200] [D loss: 0.016882] [G pixel loss: 0.170456, adv loss: 0.8616832494735718] [Elapsed time: 8475.51s]
    [Epoch 176/200] [D loss: 0.009144] [G pixel loss: 0.157934, adv loss: 0.9213641881942749] [Elapsed time: 8524.40s]
    [Epoch 177/200] [D loss: 0.009652] [G pixel loss: 0.169221, adv loss: 0.8785173296928406] [Elapsed time: 8572.41s]
    [Epoch 178/200] [D loss: 0.012084] [G pixel loss: 0.175504, adv loss: 1.0288289785385132] [Elapsed time: 8620.37s]
    [Epoch 179/200] [D loss: 0.008100] [G pixel loss: 0.192016, adv loss: 1.081184983253479] [Elapsed time: 8668.26s]
    [Epoch 180/200] [D loss: 0.018144] [G pixel loss: 0.165491, adv loss: 1.215390920639038] [Elapsed time: 8717.04s]
    [Epoch 181/200] [D loss: 0.013946] [G pixel loss: 0.154779, adv loss: 1.1482352018356323] [Elapsed time: 8765.02s]
    [Epoch 182/200] [D loss: 0.015590] [G pixel loss: 0.180115, adv loss: 0.9094292521476746] [Elapsed time: 8812.84s]
    [Epoch 183/200] [D loss: 0.070801] [G pixel loss: 0.155887, adv loss: 0.5568445920944214] [Elapsed time: 8860.85s]
    [Epoch 184/200] [D loss: 0.018930] [G pixel loss: 0.162062, adv loss: 0.8197201490402222] [Elapsed time: 8909.80s]
    [Epoch 185/200] [D loss: 0.048733] [G pixel loss: 0.164347, adv loss: 0.704258143901825] [Elapsed time: 8957.57s]
    [Epoch 186/200] [D loss: 0.014725] [G pixel loss: 0.170123, adv loss: 1.0082449913024902] [Elapsed time: 9005.52s]
    [Epoch 187/200] [D loss: 0.010266] [G pixel loss: 0.180229, adv loss: 0.9917802810668945] [Elapsed time: 9053.48s]
    [Epoch 188/200] [D loss: 0.012535] [G pixel loss: 0.163434, adv loss: 0.8837308883666992] [Elapsed time: 9102.34s]
    [Epoch 189/200] [D loss: 0.022808] [G pixel loss: 0.166665, adv loss: 0.7575783729553223] [Elapsed time: 9150.28s]
    [Epoch 190/200] [D loss: 0.010755] [G pixel loss: 0.179229, adv loss: 0.8983967304229736] [Elapsed time: 9198.24s]
    [Epoch 191/200] [D loss: 0.010936] [G pixel loss: 0.165294, adv loss: 0.9712036848068237] [Elapsed time: 9246.23s]
    [Epoch 192/200] [D loss: 0.007615] [G pixel loss: 0.156649, adv loss: 0.9319448471069336] [Elapsed time: 9295.10s]
    [Epoch 193/200] [D loss: 0.008265] [G pixel loss: 0.145452, adv loss: 0.9319097399711609] [Elapsed time: 9343.13s]
    [Epoch 194/200] [D loss: 0.013695] [G pixel loss: 0.183919, adv loss: 0.8213084936141968] [Elapsed time: 9390.98s]
    [Epoch 195/200] [D loss: 0.010004] [G pixel loss: 0.146489, adv loss: 0.9926756620407104] [Elapsed time: 9439.01s]
    [Epoch 196/200] [D loss: 0.014501] [G pixel loss: 0.154261, adv loss: 1.155073642730713] [Elapsed time: 9487.77s]
    [Epoch 197/200] [D loss: 0.012006] [G pixel loss: 0.184645, adv loss: 0.8909494876861572] [Elapsed time: 9535.80s]
    [Epoch 198/200] [D loss: 0.004491] [G pixel loss: 0.170718, adv loss: 1.0182318687438965] [Elapsed time: 9583.65s]
    [Epoch 199/200] [D loss: 0.008388] [G pixel loss: 0.154336, adv loss: 0.9526118636131287] [Elapsed time: 9631.55s]


- 생성된 이미지 예시를 출력한다.


```python
from IPython.display import Image

Image('1200.png')
```


    Output hidden; open in https://colab.research.google.com to view.


![download%20%282%29.png](attachment:download%20%282%29.png)


```python
Image('10000.png')
```


    Output hidden; open in https://colab.research.google.com to view.


![download%20%281%29.png](attachment:download%20%281%29.png)


```python
Image('2400.png')
```


    Output hidden; open in https://colab.research.google.com to view.


![download%20%283%29.png](attachment:download%20%283%29.png)

## 학습된 모델 파라미터 저장 및 테스트
 - 다음의 코드를 이용하여 학습된 모델 파라미터를 다운로드 받을 수 있다 


```python
# 모델 파라미터 저장
torch.save(generator.state_dict(), "Pix2Pix_Generator_for_Facades.pt")
torch.save(discriminator.state_dict(), "Pix2Pix_Discriminator_for_Facades.pt")
print("Model saved!")

```

    Model saved!



```python
# 모델 파라미터 다운로드
from google.colab import files

files.download('Pix2Pix_Generator_for_Facades.pt')
files.download('Pix2Pix_Discriminator_for_Facades.pt')
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


- 학습된 모델을 불러와 테스트 가능 


```python
!wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EQPOZG2WXrlHirX5l8SHM9MBVcE0pLoIAdYwd3-hWXm73Q?download=1 -O Pix2Pix_Generator_for_Facades.pt
!wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EYlJJGhQ5OFKp14jS_zGy8kBekJSkIe3ioN8LFIk59oa6A?download=1 -O Pix2Pix_Discriminator_for_Facades.pt

```

    --2022-03-21 15:32:15--  https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EQPOZG2WXrlHirX5l8SHM9MBVcE0pLoIAdYwd3-hWXm73Q?download=1
    Resolving postechackr-my.sharepoint.com (postechackr-my.sharepoint.com)... 13.107.136.9, 13.107.138.9
    Connecting to postechackr-my.sharepoint.com (postechackr-my.sharepoint.com)|13.107.136.9|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /personal/dongbinna_postech_ac_kr/Documents/Research/models/Pix2Pix/Pix2Pix_Generator_for_Facades.pt [following]
    --2022-03-21 15:32:16--  https://postechackr-my.sharepoint.com/personal/dongbinna_postech_ac_kr/Documents/Research/models/Pix2Pix/Pix2Pix_Generator_for_Facades.pt
    Reusing existing connection to postechackr-my.sharepoint.com:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 217624563 (208M) [application/octet-stream]
    Saving to: ‘Pix2Pix_Generator_for_Facades.pt’
    
    Pix2Pix_Generator_f 100%[===================>] 207.54M  53.3MB/s    in 7.0s    
    
    2022-03-21 15:32:24 (29.8 MB/s) - ‘Pix2Pix_Generator_for_Facades.pt’ saved [217624563/217624563]
    
    --2022-03-21 15:32:24--  https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EYlJJGhQ5OFKp14jS_zGy8kBekJSkIe3ioN8LFIk59oa6A?download=1
    Resolving postechackr-my.sharepoint.com (postechackr-my.sharepoint.com)... 13.107.136.9, 13.107.138.9
    Connecting to postechackr-my.sharepoint.com (postechackr-my.sharepoint.com)|13.107.136.9|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /personal/dongbinna_postech_ac_kr/Documents/Research/models/Pix2Pix/Pix2Pix_Discriminator_for_Facades.pt [following]
    --2022-03-21 15:32:25--  https://postechackr-my.sharepoint.com/personal/dongbinna_postech_ac_kr/Documents/Research/models/Pix2Pix/Pix2Pix_Discriminator_for_Facades.pt
    Reusing existing connection to postechackr-my.sharepoint.com:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 11074771 (11M) [application/octet-stream]
    Saving to: ‘Pix2Pix_Discriminator_for_Facades.pt’
    
    Pix2Pix_Discriminat 100%[===================>]  10.56M  3.29MB/s    in 3.2s    
    
    2022-03-21 15:32:29 (3.29 MB/s) - ‘Pix2Pix_Discriminator_for_Facades.pt’ saved [11074771/11074771]
    



```python
# 생성자(generator)와 판별자(discriminator) 초기화
generator = GeneratorUNet()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

generator.load_state_dict(torch.load("Pix2Pix_Generator_for_Facades.pt"))
discriminator.load_state_dict(torch.load("Pix2Pix_Discriminator_for_Facades.pt"))

generator.eval();
discriminator.eval();
```


```python
from PIL import Image

imgs = next(iter(val_dataloader)) # 10개의 이미지를 추출해 생성
real_A = imgs["B"].cuda()
real_B = imgs["A"].cuda()
fake_B = generator(real_A)
# real_A: 조건(condition), fake_B: 변환된 이미지(translated image), real_B: 정답 이미지
img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
save_image(img_sample, f"result.png", nrow=5, normalize=True)
```

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))



```python
from IPython.display import Image

Image('result.png')
```


    Output hidden; open in https://colab.research.google.com to view.


![download%20%284%29.png](attachment:download%20%284%29.png)
