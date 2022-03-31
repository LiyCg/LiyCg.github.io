---
title:  "CycleGAN - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks(2017)"
categories: DL_practice
tags:
  - Deep learning
  - DL
  - Generative Model
  - Generative Adversarial Networks
  - cycleGAN
  - unpairedDataset
toc: true
toc_sticky: true
toc_label: "CycleGAN"
toc_icon: "blog"
---




# **CycleGAN** 실습
  - 이인엽 작성
  - dldlsduq94@korea.ac.kr
  - 논문 이름: 
  Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks(2017) 


## Discriminator Class 생성


```python
import torch 
import torch.nn as nn

# 사용할 블록 만들기 
class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv = nn.Sequential(
        # "reflect": helps to reduce artifact
        nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2), # for D / ReLu for G
    )
  def forward(self, x):
    return self.conv(x)

# Discriminator 정의 
class Discriminator(nn.Module):
  def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(in_channels, 
                  features[0], 
                  kernel_size=4, 
                  stride = 2, 
                  padding=1, 
                  padding_mode="reflect"
                  ),
        nn.LeakyReLU(0.2),
    )

    layers = []
    in_channels = features[0]
    # [1:] skipping the first one becasue that was in self.initial already
    for feature in features[1:]:
      # if output feature's channel size is 512, stride's 1, otherwise it's 2
      layers.append(Block(in_channels, feature, stride= 1 if feature==features[-2] else 2 ))
      in_channels = feature
    # output returns single value(scalar), which is between 0-1, so the out_channels = 1
    layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
    # unwrapping 'layers' list to nn.Sequential  
    self.model = nn.Sequential(*layers)

    # 즉, 채널은 깊어지고, w/h는 작아진다

  def forward(self, x):
    x = self.initial(x)
    # 0-1
    return torch.sigmoid(self.model(x))


# # test if I'm outputing the right shape
# def test():
#   # in the paper, it needs to be 70X70 patch GAN!  
#   # meaning the output shape of preds.shape is 30X30 
#   # and each value in grid sees a 70X70 patch in the original image
#   x = torch.randn((5,3,256,256))
#   model = Discriminator(in_channels=3)
#   preds = model(x)
#   print(model)
#   print(preds.shape)

# if __name__ == "__main__":
#   test() 

  


 









```

## Generator Class 생성


```python

class ConvBlock(nn.Module):
  # use_act: if we use activation or not
  # kwargs: keyword arguments, can accept any parameter that exists in super init()
  def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
    super().__init__()
    self.conv = nn.Sequential(
        # ******** param 레벨에서 if / else 쓰는 법 체크 ! ********
        nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        if down
        else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True) if use_act else nn.Identity(),
    )

  def forward(self, x):
    return self.conv(x)


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()

    self.block = nn.Sequential(
        # 너비 높이 유지, (x +1*2 -3)/1 + 1 = x 니까
        ConvBlock(channels, channels, kernel_size=3, padding=1),
        # activation 안씀 
        ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
    )

  def forward(self, x):
    # concatenating previous low-level info
    return  x + self.block(x)



class Generator(nn.Module):

  # 256X256 이상의 shape이라면 num_residuals=9
  #         미만의 shape이라면 num_residuals=6
  def __init__(self, img_channels, num_features=64, num_residuals=9):
    super().__init__()

    # won't use intancenorm 
    self.initial = nn.Sequential(
         nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
         nn.ReLU(inplace=True),
    )

    self.down_blocks = nn.ModuleList(
        [
         # down=True by default
         ConvBlock(num_features,num_features*2, kernel_size=3, stride = 2, padding=1),
         ConvBlock(num_features*2,num_features*4, kernel_size=3, stride = 2, padding=1),
        ]
    )

    # residual blocks don't change the input/num of channels
    # nn.Sequential은 List를 언랩한 것을 받을 수 있음
    self.residual_blocks = nn.Sequential(
        # bunch of residuals ..
        # '_' meaning there's no iterable object specified, just wanna iterate
        *[ResidualBlock(num_features*4) for  _ in range(num_residuals)] 
    )

    self.up_blocks = nn.ModuleList(
        [
        # output_padding: additional padding after the convBlock 
        # it's not neccesary in the most case (following what the paper did)
        ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride = 2, padding=1, output_padding = 1),
        # 출력 channel: 64
        ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride = 2, padding=1, output_padding = 1),
        ]
    )

    # 아직 channel이 RGB 즉, 3이 아니므로, Img_channels로 바꿔줌
    self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode = "reflect")



  def forward(self, x):

    x = self.initial(x)
    
    for layer in self.down_blocks:
      x = layer(x)
    
    x = self.residual_blocks(x)
    
    for layer in self.up_blocks:
      x = layer(x)
    
    # -1 ~ 1 range
    return torch.tanh(self.last(x))


# # test if I'm outputing the right shape
# def test():
#   img_channels = 3
#   img_size = 256
#   x = torch.randn((2,img_channels,256,256))
#   gen = Generator(img_channels,9)
#   print(gen)
#   print(gen(x).shape) # 2X3X256X256

# if __name__ == "__main__":
#   test() 


```

## 데이터 로드

  - 데이터 셋 다운로드


```python
# cloning repository
# 한번 다운로드 했으므로 이제 실행할 필요 없다
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```

    Cloning into 'pytorch-CycleGAN-and-pix2pix'...
    remote: Enumerating objects: 2447, done.[K
    remote: Total 2447 (delta 0), reused 0 (delta 0), pack-reused 2447[K
    Receiving objects: 100% (2447/2447), 8.18 MiB | 23.72 MiB/s, done.
    Resolving deltas: 100% (1535/1535), done.



```python
# mounting drive
from google.colab import drive
drive.mount('/content/drive/',force_remount=True) 
```

    Mounted at /content/drive/



```python
import os

os.chdir('drive/MyDrive/pytorch-CycleGAN-and-pix2pix/')
```


```python
# downloading requirements
!pip install -r requirements.txt
```

    Requirement already satisfied: torch>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (1.10.0+cu111)
    Requirement already satisfied: torchvision>=0.5.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 2)) (0.11.1+cu111)
    Collecting dominate>=2.4.0
      Downloading dominate-2.6.0-py2.py3-none-any.whl (29 kB)
    Collecting visdom>=0.1.8.8
      Downloading visdom-0.1.8.9.tar.gz (676 kB)
    [K     |████████████████████████████████| 676 kB 17.8 MB/s 
    [?25hCollecting wandb
      Downloading wandb-0.12.11-py2.py3-none-any.whl (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 57.8 MB/s 
    [?25hRequirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.4.0->-r requirements.txt (line 1)) (3.10.0.2)
    Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision>=0.5.0->-r requirements.txt (line 2)) (7.1.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torchvision>=0.5.0->-r requirements.txt (line 2)) (1.21.5)
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
      Downloading websocket_client-1.3.2-py3-none-any.whl (54 kB)
    [K     |████████████████████████████████| 54 kB 3.4 MB/s 
    [?25hCollecting yaspin>=1.0.0
      Downloading yaspin-2.1.0-py3-none-any.whl (18 kB)
    Requirement already satisfied: promise<3,>=2.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (2.3)
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (5.4.8)
    Collecting GitPython>=1.0.0
      Downloading GitPython-3.1.27-py3-none-any.whl (181 kB)
    [K     |████████████████████████████████| 181 kB 33.8 MB/s 
    [?25hCollecting sentry-sdk>=1.0.0
      Downloading sentry_sdk-1.5.8-py2.py3-none-any.whl (144 kB)
    [K     |████████████████████████████████| 144 kB 68.6 MB/s 
    [?25hCollecting setproctitle
      Downloading setproctitle-1.2.2-cp37-cp37m-manylinux1_x86_64.whl (36 kB)
    Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (2.8.2)
    Collecting pathtools
      Downloading pathtools-0.1.2.tar.gz (11 kB)
    Requirement already satisfied: Click!=8.0.0,>=7.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (7.1.2)
    Requirement already satisfied: protobuf>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (3.17.3)
    Collecting shortuuid>=0.5.0
      Downloading shortuuid-1.0.8-py3-none-any.whl (9.5 kB)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from wandb->-r requirements.txt (line 5)) (3.13)
    Collecting docker-pycreds>=0.4.0
      Downloading docker_pycreds-0.4.0-py2.py3-none-any.whl (9.0 kB)
    Collecting gitdb<5,>=4.0.1
      Downloading gitdb-4.0.9-py3-none-any.whl (63 kB)
    [K     |████████████████████████████████| 63 kB 2.2 MB/s 
    [?25hCollecting smmap<6,>=3.0.1
      Downloading smmap-5.0.0-py3-none-any.whl (24 kB)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (2021.10.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->visdom>=0.1.8.8->-r requirements.txt (line 4)) (2.10)
    Requirement already satisfied: termcolor<2.0.0,>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from yaspin>=1.0.0->wandb->-r requirements.txt (line 5)) (1.1.0)
    Collecting jsonpointer>=1.9
      Downloading jsonpointer-2.2-py2.py3-none-any.whl (7.5 kB)
    Building wheels for collected packages: visdom, pathtools, torchfile
      Building wheel for visdom (setup.py) ... [?25l[?25hdone
      Created wheel for visdom: filename=visdom-0.1.8.9-py3-none-any.whl size=655250 sha256=399ca9f9d02a2cafbc470237e66710a5203cc90345ce4e70d5b2d04576da98e7
      Stored in directory: /root/.cache/pip/wheels/2d/d1/9b/cde923274eac9cbb6ff0d8c7c72fe30a3da9095a38fd50bbf1
      Building wheel for pathtools (setup.py) ... [?25l[?25hdone
      Created wheel for pathtools: filename=pathtools-0.1.2-py3-none-any.whl size=8806 sha256=7bd4e5b43a10392962967102c0afdf1835131f1e9ea885238849cb212005db69
      Stored in directory: /root/.cache/pip/wheels/3e/31/09/fa59cef12cdcfecc627b3d24273699f390e71828921b2cbba2
      Building wheel for torchfile (setup.py) ... [?25l[?25hdone
      Created wheel for torchfile: filename=torchfile-0.1.0-py3-none-any.whl size=5709 sha256=f43c535f0fcd7398203809f057278349e96792c0ec54cc1b4fb60ffd0cc7a3b7
      Stored in directory: /root/.cache/pip/wheels/ac/5c/3a/a80e1c65880945c71fd833408cd1e9a8cb7e2f8f37620bb75b
    Successfully built visdom pathtools torchfile
    Installing collected packages: smmap, jsonpointer, gitdb, yaspin, websocket-client, torchfile, shortuuid, setproctitle, sentry-sdk, pathtools, jsonpatch, GitPython, docker-pycreds, wandb, visdom, dominate
    Successfully installed GitPython-3.1.27 docker-pycreds-0.4.0 dominate-2.6.0 gitdb-4.0.9 jsonpatch-1.32 jsonpointer-2.2 pathtools-0.1.2 sentry-sdk-1.5.8 setproctitle-1.2.2 shortuuid-1.0.8 smmap-5.0.0 torchfile-0.1.0 visdom-0.1.8.9 wandb-0.12.11 websocket-client-1.3.2 yaspin-2.1.0


- Datasets

  Download one of the official datasets with:

-   `bash ./datasets/download_cyclegan_dataset.sh [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos]`

- Or use your own dataset by creating the appropriate folders and adding in the images.

-  Create a dataset folder under `/dataset` for your dataset.
-   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the `testA` folder, images you want to transform from b to a (dog2cat) in the `testB` folder, and do the same for the `trainA` and `trainB` folders.

  - 학습 속도 높이기위해 구글 드라이브 직접접근 막기 
    - 데이터 셋을 /content로 옮겨온다


```python
import os

# Location of Zip File
drive_path = '/content/drive/MyDrive/pytorch-CycleGAN-and-pix2pix/horse2zebra-20220327T123200Z-001.zip'
# colab instance(local)의 path
local_path = '/content'

# Copy the zip file and move it up one level (AKA out of the drive folder)
!cp '{drive_path}' .

# Navigate to the copied file and unzip it quietly
os.chdir(local_path)
!unzip -q 'horse2zebra-20220327T123200Z-001.zip'
```


```python
# # dataset download (horse2zebra only)
# # now downloaded permenantly at MyDrive, not neccessary
# !bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

    Specified [horse2zebra]
    WARNING: timestamping does nothing in combination with -O. See the manual
    for details.
    
    --2022-03-26 14:52:39--  http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
    Resolving efrosgans.eecs.berkeley.edu (efrosgans.eecs.berkeley.edu)... 128.32.244.190
    Connecting to efrosgans.eecs.berkeley.edu (efrosgans.eecs.berkeley.edu)|128.32.244.190|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 116867962 (111M) [application/zip]
    Saving to: ‘./datasets/horse2zebra.zip’
    
    ./datasets/horse2ze 100%[===================>] 111.45M  3.17MB/s    in 49s     
    
    2022-03-26 14:53:28 (2.30 MB/s) - ‘./datasets/horse2zebra.zip’ saved [116867962/116867962]
    
    Archive:  ./datasets/horse2zebra.zip
       creating: ./datasets/horse2zebra/trainA/
      inflating: ./datasets/horse2zebra/trainA/n02381460_6223.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1567.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3354.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_299.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3001.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4242.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1666.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4396.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4502.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8527.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_14.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_706.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4019.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1478.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3449.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5558.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_969.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1494.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1435.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5927.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_979.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4621.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2412.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4474.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5803.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_276.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1432.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3353.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5538.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2198.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4347.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6944.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4373.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2751.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6089.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2755.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4276.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_322.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3221.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_855.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4141.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_807.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1019.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1319.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2341.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9064.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1939.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1208.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_993.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_863.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1586.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4425.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1226.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3438.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_948.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_736.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1197.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4429.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3754.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1482.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6402.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5805.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2462.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_357.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2476.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1862.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5983.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5944.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2985.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3897.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1105.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8759.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_732.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4729.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3548.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3369.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7515.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1658.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1075.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1501.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8903.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_674.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4687.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4566.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_255.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_908.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1204.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_236.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7689.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_155.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2048.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3601.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1675.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1888.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8325.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4413.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_186.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8735.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9083.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1489.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1874.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3085.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4763.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3483.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7512.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2223.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_668.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_596.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9234.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3393.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6289.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_986.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8668.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2446.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1001.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1624.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3412.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1122.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3175.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4374.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8639.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_528.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_11.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3984.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1135.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3902.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_199.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2967.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_677.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2043.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1337.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1769.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2325.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2032.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_27.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7261.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_852.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5287.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2103.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8944.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7122.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1562.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1365.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_856.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6549.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7495.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_108.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3066.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5442.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1796.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_643.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1487.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1025.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5618.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2578.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8306.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_105.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4748.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4944.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1952.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5721.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_919.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2779.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1422.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7905.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_998.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6743.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2929.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_697.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7876.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_122.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3315.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3946.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1813.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1941.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_242.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_381.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6441.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4701.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4629.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5525.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2014.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3299.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1006.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2061.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7695.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1009.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2793.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1347.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2027.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_591.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6964.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8769.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2449.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4467.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1227.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3424.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_249.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_274.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1288.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1506.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_718.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2649.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5292.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4648.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3365.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4613.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5699.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1901.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1058.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3664.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2758.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3081.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5266.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_903.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3417.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_869.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6204.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_518.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8725.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4645.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3115.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1751.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8241.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3619.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2425.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5241.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9127.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5869.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2484.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3192.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4372.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6294.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7456.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1439.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_754.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4219.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4403.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4118.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4465.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_69.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4674.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_175.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6397.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_386.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4491.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5036.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_503.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8895.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1023.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8157.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4453.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1262.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2402.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2179.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1034.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4417.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_911.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6556.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1038.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_782.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1276.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3447.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6981.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_525.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_331.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1979.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3512.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5087.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1402.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2268.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1387.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2016.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1405.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2465.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_476.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1246.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1003.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1484.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2741.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_671.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1155.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_811.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_739.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2089.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_422.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2706.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1048.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8714.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1132.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5502.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5607.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6873.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9052.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1216.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1665.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_616.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3712.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9243.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2177.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4472.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4991.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1267.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3212.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1519.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2795.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1803.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_222.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3358.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3113.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2627.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5696.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4473.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3018.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6476.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_238.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_627.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3283.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_543.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1027.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4771.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3982.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6792.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4817.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4294.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2339.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6466.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2093.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1651.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5704.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7412.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2936.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4917.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2349.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3694.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_36.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4779.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3942.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1856.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_61.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1614.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6782.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3707.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_541.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_728.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4642.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4018.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9063.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2245.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2563.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7932.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_454.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4551.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2643.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1084.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3911.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1596.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_396.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5406.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8974.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1297.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4059.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4788.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8838.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2374.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7537.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4856.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5029.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4081.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3075.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_411.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4008.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1579.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4815.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_203.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3509.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_631.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_196.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7597.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1713.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2109.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2317.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2753.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4515.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3783.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1578.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1978.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2684.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1458.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_967.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2139.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2019.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_558.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3905.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1581.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2327.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1687.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6987.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2023.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2822.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5708.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3854.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2953.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1521.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5864.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1373.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7919.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_477.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8006.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1045.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3107.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1852.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_125.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4493.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_235.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1507.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2419.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_891.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_521.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5526.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1721.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4999.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1758.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_66.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2517.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1232.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2474.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2259.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2565.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4079.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_391.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3363.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2192.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8437.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1426.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2468.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_968.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4523.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5857.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3661.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5214.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4711.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4513.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4928.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7231.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4927.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5203.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8985.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5411.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2552.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4231.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6152.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2995.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8048.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1191.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2714.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1044.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4665.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4783.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_703.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_278.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2596.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7476.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7297.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4516.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5891.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7571.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6403.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1305.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1991.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1798.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_595.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2083.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_921.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2555.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_858.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4625.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8277.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3366.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2695.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3265.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4053.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_804.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4107.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1995.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3336.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2403.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_424.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4726.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_489.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2605.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1331.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4112.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7803.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2921.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_584.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4402.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3557.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4946.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2615.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_117.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3685.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_854.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6388.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2883.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1583.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_803.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1973.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1525.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5045.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4608.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1703.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1266.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3746.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_971.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4125.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7493.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4538.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8088.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6978.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2217.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1736.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_394.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_757.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7348.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_935.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1711.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_881.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4371.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4593.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4153.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4536.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4785.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4401.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3065.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4528.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_933.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3346.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5355.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4965.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6975.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8013.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3383.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2796.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3546.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_437.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4239.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4009.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_755.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5265.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4476.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5729.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8435.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4117.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1249.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1063.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1916.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8179.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1144.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6876.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6277.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2632.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_788.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3329.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_223.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2799.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2528.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4624.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2499.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1526.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1307.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7942.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5489.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1865.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1648.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1592.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2282.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_135.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4251.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2149.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1258.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1872.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1486.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1612.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1108.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7845.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2305.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3394.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4585.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3924.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1904.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7548.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3149.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5551.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4638.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5892.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1621.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3228.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3551.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4637.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6268.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7126.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1734.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_825.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6721.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1068.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1391.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4086.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7806.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4967.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2226.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3398.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2811.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3811.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_306.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4501.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_177.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1127.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7528.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1835.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3702.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2598.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4661.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2575.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8877.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_767.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1537.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4385.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2581.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7858.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3875.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1635.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1292.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1743.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1388.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7651.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_153.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_897.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_506.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4494.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2433.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_707.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4094.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6946.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_565.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3995.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_384.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2525.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_624.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1915.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5167.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4263.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4305.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3656.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7756.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1847.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_743.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1602.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2754.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3829.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1423.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2294.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_195.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4146.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_182.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2289.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4731.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2594.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6411.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6836.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4168.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_358.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4606.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3051.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8507.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3211.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_902.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3997.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4338.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_323.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8389.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1283.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8575.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1348.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1106.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1645.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3566.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7779.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1384.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_393.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1981.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2269.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_879.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6822.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1498.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3732.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_419.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7622.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7178.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8586.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6761.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_588.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1909.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1957.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2056.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1074.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1418.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_451.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7692.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6297.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_764.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1722.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6335.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3229.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1834.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2703.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3861.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_314.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3108.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3563.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2411.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2366.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5537.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1763.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1959.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1389.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1134.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8104.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4804.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1509.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2421.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3371.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2141.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1696.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1014.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1338.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3182.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1727.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3556.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1052.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8659.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_834.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1336.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1524.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8704.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7613.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1008.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5782.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8765.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_559.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5674.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_941.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5156.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7165.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3956.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3348.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_421.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4482.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4497.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2276.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4395.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2261.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1683.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3286.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2783.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_285.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9095.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8515.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7959.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2116.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9263.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_776.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2488.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3989.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_777.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2872.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5519.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3577.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1533.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5396.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2414.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5915.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6373.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2635.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_598.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1637.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1855.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2275.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4397.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_676.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4542.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2834.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2509.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1905.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4521.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2835.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1083.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4809.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1158.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2091.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_461.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3965.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4055.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3659.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4151.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7634.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1002.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3091.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4446.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2685.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1184.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3928.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5819.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8862.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1101.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2548.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3611.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6258.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3469.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2045.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4668.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1028.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_102.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1816.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_567.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3838.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_654.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4456.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_474.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1921.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1011.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1159.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6995.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_545.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2503.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_398.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1493.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2968.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_395.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4407.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4916.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5374.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_178.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_742.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4657.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_311.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4743.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5386.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4412.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4751.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_452.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_211.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3526.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9184.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1182.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_128.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1674.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4572.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2699.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3309.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4415.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5878.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_901.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4169.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4123.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1557.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2878.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5075.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4103.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5601.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4632.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1098.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7888.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6252.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_58.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8153.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_24.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4598.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4597.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7398.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3935.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1708.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3466.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_747.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4833.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1247.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1563.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6941.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_594.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_601.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1147.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_726.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2444.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3857.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1801.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8626.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3638.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1354.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_849.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3442.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4781.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1051.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4136.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8989.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9223.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3903.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4331.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4975.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1774.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_691.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3842.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2708.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4159.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4195.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6511.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2788.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4483.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1427.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_202.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_553.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5135.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2306.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2806.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4096.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1992.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4228.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1672.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7754.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4602.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7305.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4361.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_976.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5761.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_981.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3588.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7219.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1573.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4639.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2181.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2879.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4571.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_466.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5951.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2848.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2152.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4181.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1766.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5545.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4737.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4766.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5917.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7903.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2572.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_328.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1082.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_939.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2612.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_519.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1425.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4543.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_912.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_582.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1682.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2012.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1361.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5107.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1686.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3355.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6824.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4541.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_769.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2085.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8708.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_509.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8718.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5771.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4816.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1639.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2514.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_836.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5804.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_86.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3028.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8449.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4376.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1793.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3395.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2736.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1053.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3803.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_552.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_798.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1035.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3255.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6922.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8611.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5739.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6267.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1334.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8812.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_73.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1168.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4447.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_778.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4324.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1037.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4348.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1523.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1531.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5408.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2496.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4695.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8479.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3222.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4307.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5034.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4434.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8577.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1804.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1937.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2951.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1761.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1212.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4977.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2539.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8958.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3605.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_629.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_765.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1263.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3226.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1194.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_91.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_5286.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4769.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4312.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2222.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2049.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8052.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4663.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1349.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4529.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_8585.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2726.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_9167.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2732.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2802.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6688.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_501.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2371.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_835.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_675.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3827.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4758.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_1363.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3064.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2564.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_4084.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_376.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_717.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_784.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_2225.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_7033.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_6763.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_3658.jpg  
      inflating: ./datasets/horse2zebra/trainA/n02381460_19.jpg  
       creating: ./datasets/horse2zebra/testB/
      inflating: ./datasets/horse2zebra/testB/n02391049_3130.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4110.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_640.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_260.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_7060.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2200.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2500.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1270.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5220.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3220.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_120.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4610.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9460.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10980.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1760.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2470.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2800.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3070.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_980.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2190.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1290.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10590.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1150.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9350.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10910.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1220.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1430.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6650.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2510.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_790.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_490.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_100.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1300.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10210.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2600.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1950.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_900.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_270.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2420.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1340.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_890.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2620.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_600.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4490.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9900.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9400.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6860.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_440.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2970.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2220.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1020.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2380.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3800.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6180.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3060.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_7740.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1880.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4570.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2810.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6890.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5930.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10100.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3770.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_410.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1100.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4730.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6190.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_390.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6520.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3840.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_480.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8340.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_750.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1000.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1630.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_430.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6780.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_400.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5670.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3290.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8000.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_80.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1790.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_690.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10810.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2290.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2350.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3240.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3310.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2570.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4990.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9960.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3320.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3450.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_7190.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_7150.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_6690.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2100.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8020.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8830.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2730.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_920.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8140.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3750.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_7860.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_180.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_1060.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_170.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_4890.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10160.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_10630.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5100.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5240.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_200.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3090.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9160.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2930.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2890.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2410.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_8080.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5720.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5320.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9740.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3010.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_560.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9000.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3200.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2990.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5810.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2480.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_3270.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2460.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5030.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_9680.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_5990.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_130.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2790.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2760.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_860.jpg  
      inflating: ./datasets/horse2zebra/testB/n02391049_2870.jpg  
       creating: ./datasets/horse2zebra/trainB/
      inflating: ./datasets/horse2zebra/trainB/n02391049_155.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3211.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_295.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4692.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8178.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8032.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2709.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2418.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2985.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2612.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2049.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1087.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2476.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2285.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9149.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2232.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7722.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_737.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5509.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_589.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2211.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8061.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5875.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1042.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2446.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2157.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_141.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6184.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_344.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2989.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2005.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2116.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_354.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2999.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_465.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2789.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1104.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7407.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2424.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_415.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10467.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2776.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5366.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8447.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4697.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2628.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1869.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1158.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11162.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11195.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3239.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2698.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9675.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2048.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9804.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_544.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2984.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6121.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_744.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_839.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2017.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4569.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3402.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1758.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2289.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_817.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_527.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_56.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6122.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7886.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2541.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2559.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3135.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2816.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5781.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10589.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5982.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_176.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_116.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6511.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10278.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6336.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2131.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7263.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9977.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_351.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9357.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6993.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2921.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5551.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4028.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_254.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_77.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2886.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3092.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3733.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2739.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6524.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_934.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_835.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4775.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6026.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_826.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4077.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_886.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2061.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8764.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1149.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7972.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2889.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3328.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2043.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_554.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4058.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9416.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3654.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2383.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3261.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2907.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_169.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1928.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6902.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_808.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10132.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2511.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2459.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_717.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3071.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4185.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10824.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10497.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6081.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3242.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_437.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2657.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_168.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8856.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_279.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_225.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2821.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7836.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2366.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_71.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_368.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2304.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_341.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_115.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10701.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3778.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_444.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7261.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9136.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_46.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2648.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2644.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_275.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2673.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1752.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_644.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6012.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3199.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1894.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4969.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3229.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1738.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_246.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3872.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6198.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8444.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1679.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7832.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_795.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2239.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_178.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1751.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5695.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8387.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9051.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9593.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11181.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_173.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7494.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5503.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_248.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9719.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1531.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3022.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8699.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3779.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2806.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4898.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_363.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3173.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3681.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_904.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9398.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_232.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2201.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3947.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7839.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4259.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_226.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1891.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2534.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6834.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_643.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8565.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2182.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2837.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9918.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2718.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6326.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10429.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2508.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_383.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6267.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8347.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6374.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10649.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1532.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2856.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_147.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1126.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7398.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8568.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1076.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6202.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7062.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_308.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6436.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4805.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10681.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10738.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_17.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_935.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2927.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3517.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2364.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_764.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2665.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3084.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8425.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_983.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1731.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1451.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1073.jpg  
     extracting: ./datasets/horse2zebra/trainB/n02391049_7503.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_712.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9822.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10948.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7567.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1178.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5638.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3397.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8857.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2204.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3296.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10917.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2606.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2916.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4745.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8659.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2995.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1172.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3228.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2275.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_316.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_499.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1296.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7396.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6452.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11166.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6421.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_907.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2701.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3292.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10825.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2844.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1088.jpg  
     extracting: ./datasets/horse2zebra/trainB/n02391049_2361.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_192.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1316.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3073.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8214.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2187.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7169.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_19.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1056.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3436.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_574.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_639.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_133.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2596.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3016.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2732.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_538.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1153.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3693.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1147.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1156.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_734.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_956.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9113.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_384.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_695.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5209.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5115.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8859.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3115.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1121.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_635.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3518.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2638.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2162.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_459.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2251.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_343.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2899.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2181.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8793.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9542.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3023.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2786.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_23.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9273.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_274.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_824.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10134.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2847.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3248.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_789.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_125.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3785.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2321.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_964.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1836.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5791.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2982.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6762.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3233.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_474.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1024.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10461.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_557.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4027.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3038.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3755.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7151.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2811.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1523.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1112.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4496.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6015.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10257.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3209.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2603.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10307.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1025.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_228.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8373.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3631.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3257.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_865.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9193.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1303.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1465.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2098.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10739.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3901.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3303.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8657.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2504.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_419.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3636.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9151.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2506.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7599.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1189.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6163.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2222.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_427.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9964.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6467.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9635.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5217.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8015.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_193.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2591.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3028.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2507.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5344.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3046.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2295.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2355.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4824.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_377.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10432.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1164.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1288.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2523.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10159.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2924.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2242.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3001.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_428.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3465.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_182.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3754.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5307.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3342.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6475.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5547.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2293.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6947.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9006.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2645.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3154.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1459.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_217.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_33.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10324.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6866.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8844.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_954.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2265.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10269.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3236.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_424.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2623.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5595.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2768.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3235.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7487.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3128.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9698.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4517.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8951.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3348.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_515.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_566.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_37.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6465.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2407.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4176.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_26.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4474.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2273.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3861.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_642.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_98.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9336.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_986.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_148.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7105.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3075.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6396.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2996.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6444.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9919.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_142.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_471.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_711.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_936.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3105.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2723.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_543.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5501.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7188.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2133.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2852.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7911.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10576.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2481.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8028.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3153.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2946.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_608.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3938.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1818.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_251.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_556.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10175.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10342.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1743.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_514.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8075.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8209.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2024.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2627.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7668.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2549.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6565.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1817.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4952.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1054.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3174.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_154.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1509.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2465.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_79.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4974.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8878.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4914.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2416.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2687.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2696.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3044.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_329.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2451.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4748.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_743.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8539.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2485.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5291.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9494.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3289.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2155.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10596.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3148.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1046.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_43.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_32.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1095.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3958.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5371.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6655.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7096.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3202.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_487.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8291.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6503.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2218.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3306.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8526.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5679.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1973.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6804.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2819.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2731.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5788.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4161.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_927.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_35.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1236.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7115.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_796.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2748.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_283.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2036.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3114.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2228.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6818.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1053.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6074.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4329.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6023.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_628.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7368.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_267.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1295.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7589.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5273.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1652.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2805.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_949.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3371.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4863.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3844.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10591.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3339.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2166.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_332.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_915.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6293.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7648.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4644.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1118.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2136.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2057.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3905.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2104.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1946.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_533.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5909.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2034.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6053.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3276.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_909.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6951.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2177.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10158.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6386.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_614.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5189.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2495.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6918.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10398.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1124.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_272.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7418.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_261.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_745.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2656.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_119.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_757.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9421.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2571.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9922.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8201.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10636.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6208.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1908.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_721.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2486.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_638.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2967.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1135.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_164.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3293.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9487.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9443.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7563.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2214.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_234.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4932.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1436.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4351.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1434.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8782.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_851.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_475.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1551.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2926.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3139.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3949.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_944.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1113.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2963.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6469.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3415.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3264.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_345.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_468.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3041.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3034.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9377.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1127.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2836.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1223.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1028.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2146.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3446.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10244.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10027.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1159.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3536.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3291.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5383.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2163.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1012.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_596.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_966.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2466.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3271.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7048.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6631.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1256.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4576.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_633.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1555.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_752.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_929.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_976.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8008.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2229.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_425.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_74.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1725.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9135.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6245.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_496.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_718.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10427.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10339.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5908.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1937.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10209.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9911.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5924.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3937.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10295.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2782.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_925.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2794.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2895.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1004.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_834.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1026.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7466.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7318.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2784.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_333.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5121.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6236.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2457.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_733.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3098.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1479.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1721.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2997.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4704.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5131.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_286.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_58.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_686.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11128.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1163.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_738.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6413.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8148.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_704.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_276.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9519.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1524.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7319.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4638.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_897.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3085.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2141.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9497.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2478.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6683.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_39.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8301.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_171.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1038.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1417.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9532.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5637.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4677.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2464.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2785.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7851.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7581.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2883.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2781.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3079.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_146.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9062.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_121.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10966.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_314.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6957.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_659.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3365.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6564.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9365.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_788.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5208.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2637.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5985.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2671.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2922.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1992.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2714.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_36.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9441.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3087.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7324.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1188.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_497.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9844.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2111.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4873.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_541.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2818.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_626.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3428.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6922.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_463.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2853.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2825.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7584.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_167.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_152.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4889.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2651.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9427.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3072.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10007.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2341.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3917.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6546.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6582.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4448.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2674.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1309.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2379.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2636.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4747.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9309.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7038.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9656.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_921.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2584.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_932.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_353.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9965.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2758.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1267.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3031.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2659.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2857.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_707.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7512.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_985.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_536.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4682.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_306.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2681.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2333.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_328.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7018.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1852.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_924.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6385.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3426.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7927.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6471.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9478.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2741.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2363.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10322.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2957.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_101.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_87.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1211.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1105.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2663.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_776.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4292.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7645.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3578.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2408.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2258.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1464.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7919.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3081.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3221.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_627.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9747.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1535.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_978.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2824.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6216.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4693.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6159.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_787.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2524.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11041.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6097.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2579.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_578.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4069.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10047.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2356.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1099.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2941.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4525.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9705.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_841.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1875.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8192.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_999.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2184.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1441.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4901.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4896.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1442.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_45.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6487.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3238.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9499.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2191.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2448.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4034.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5853.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10597.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9524.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7071.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2871.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3749.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10356.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2831.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_993.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_285.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9137.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_719.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2798.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2367.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_629.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_773.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4475.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8488.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2639.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3509.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1314.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2281.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2183.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6962.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2838.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7566.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3253.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2633.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2973.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5295.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6227.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8334.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1174.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3266.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3868.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5633.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4903.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3091.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2879.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7457.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_63.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9979.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_407.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5169.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2593.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2003.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3251.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3333.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6622.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3771.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2964.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8982.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8021.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4965.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3326.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9528.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2877.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1404.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5527.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9864.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1191.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2906.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1064.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1148.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2901.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2231.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1526.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_873.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2597.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1086.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2684.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_725.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8616.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_685.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5782.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_403.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_601.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1846.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6664.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_715.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_483.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2959.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9533.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5789.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6629.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3513.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10063.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8689.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2829.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7683.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_22.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_109.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_406.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_602.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2911.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2691.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3086.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4189.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1199.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10435.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2386.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_786.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8994.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4879.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4734.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2372.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8593.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7258.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_268.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_699.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2905.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10131.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2788.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1063.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6935.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2091.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2875.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9055.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7979.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5618.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2766.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8394.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4839.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6735.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2477.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_85.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2616.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1145.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_857.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3294.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_916.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3324.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_396.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9629.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9626.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5586.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7409.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_108.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1952.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1181.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7416.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1789.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9726.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2803.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1131.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3267.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2817.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_404.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_421.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7236.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8288.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1567.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6517.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7721.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_984.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2747.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2749.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3053.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8831.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_392.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3096.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7306.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2588.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_611.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2935.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2014.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_918.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1184.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_957.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2585.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2888.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3026.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5276.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_161.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10336.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2165.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_501.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3069.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6775.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2405.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1827.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10123.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7847.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2876.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3177.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1195.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10682.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1129.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_675.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3162.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2469.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2631.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2609.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5714.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1742.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10749.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2018.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1855.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_621.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2221.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2702.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7378.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5755.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2437.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2752.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_166.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2337.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_163.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9583.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7557.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_676.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_943.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1097.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10129.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_41.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10754.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1355.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6317.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3319.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3805.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9625.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_587.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2518.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9951.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9317.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_713.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2322.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_127.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6856.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1378.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_605.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3317.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2693.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1419.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8855.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1125.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_25.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_913.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9058.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3187.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_551.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_18.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2771.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_809.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2491.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7364.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4261.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_221.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1884.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3137.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5661.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6919.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2297.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4814.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4445.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3314.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3232.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10122.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_847.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4584.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10239.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3088.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1495.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3158.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_969.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_461.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8158.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9303.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9016.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3149.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2286.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9331.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3262.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9417.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5947.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1844.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6378.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5889.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10837.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10454.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3443.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2733.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3249.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7812.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2756.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4144.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6186.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_249.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8633.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_102.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2375.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8059.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9762.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10504.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_926.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_992.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3043.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6609.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_682.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_224.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2197.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2974.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4413.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_362.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_335.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3623.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2897.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3191.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2266.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1235.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5216.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_358.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5724.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7344.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9219.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7949.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3582.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2642.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_646.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1712.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_866.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2439.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_107.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7848.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3122.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2746.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8182.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2647.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_12.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3054.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8902.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2848.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2381.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6359.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2223.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6561.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8266.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2767.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9387.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7074.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3246.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4528.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6757.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6987.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11153.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7077.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_565.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_603.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4729.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2951.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3682.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3018.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10613.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_775.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_11063.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_671.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2729.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9002.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4922.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4909.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_122.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2949.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2743.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2704.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_395.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_357.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2842.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2757.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_15.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9555.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1175.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_812.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8321.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5711.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10452.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1183.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9987.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3878.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9777.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4564.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7779.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8149.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5093.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_879.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2311.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_379.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2567.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_111.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_356.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5356.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5367.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1067.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2586.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3219.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5236.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_10426.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1516.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_584.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2814.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8102.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3037.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6512.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3265.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_968.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5333.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1062.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2306.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7678.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2826.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5015.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7434.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2225.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_506.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_5005.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_1198.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9122.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7205.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_875.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_118.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6991.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_202.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3024.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9556.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2117.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8445.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8696.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9027.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9258.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_7521.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_8937.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4218.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_9665.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2294.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_6942.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2813.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_4106.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3247.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_2217.jpg  
      inflating: ./datasets/horse2zebra/trainB/n02391049_3119.jpg  
       creating: ./datasets/horse2zebra/testA/
      inflating: ./datasets/horse2zebra/testA/n02381460_1110.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1740.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1300.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1260.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4410.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4260.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1690.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_9240.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_670.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4430.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4110.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6920.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_20.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4010.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2950.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3120.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2050.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2150.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_5090.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1750.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6300.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7190.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3660.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_5500.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1100.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2540.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6640.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2890.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4630.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7140.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7250.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2120.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4160.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_900.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1350.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2870.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_360.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_8900.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_500.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_440.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1540.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7300.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3240.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4640.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7170.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_9260.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1000.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_140.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4470.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7620.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1210.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7660.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4370.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1830.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_5940.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3910.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4450.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2940.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2100.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6950.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4530.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1360.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_510.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2710.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4550.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2580.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4650.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3330.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_470.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1870.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_490.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3040.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2460.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_640.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4800.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2650.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4740.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_950.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_180.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6790.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1010.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1920.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4790.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_2280.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_530.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_40.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_8980.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1630.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6290.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3110.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_50.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1620.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1160.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1090.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_600.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1120.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_690.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7890.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7700.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7400.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_200.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_6690.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_3010.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7230.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1030.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_910.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_5670.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4420.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_800.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4660.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1420.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_840.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4120.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_120.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1820.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_1660.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4310.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7500.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_4240.jpg  
      inflating: ./datasets/horse2zebra/testA/n02381460_7970.jpg  


- albumentations 패키지 downloading
- colab에 없음(pycharm에서는 필요없을듯) 


```python
# downloading ToTensorV2
!pip install albumentations==0.4.6
```

    Collecting albumentations==0.4.6
      Downloading albumentations-0.4.6.tar.gz (117 kB)
    [K     |████████████████████████████████| 117 kB 34.0 MB/s 
    [?25hRequirement already satisfied: numpy>=1.11.1 in /usr/local/lib/python3.7/dist-packages (from albumentations==0.4.6) (1.21.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from albumentations==0.4.6) (1.4.1)
    Collecting imgaug>=0.4.0
      Downloading imgaug-0.4.0-py2.py3-none-any.whl (948 kB)
    [K     |████████████████████████████████| 948 kB 31.5 MB/s 
    [?25hRequirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from albumentations==0.4.6) (3.13)
    Requirement already satisfied: opencv-python>=4.1.1 in /usr/local/lib/python3.7/dist-packages (from albumentations==0.4.6) (4.1.2.30)
    Requirement already satisfied: imageio in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (2.4.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (3.2.2)
    Requirement already satisfied: Shapely in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (1.8.1.post1)
    Requirement already satisfied: scikit-image>=0.14.2 in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (0.18.3)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (1.15.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (7.1.2)
    Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (1.3.0)
    Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (2021.11.2)
    Requirement already satisfied: networkx>=2.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (2.6.3)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (1.4.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (3.0.7)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (3.10.0.2)
    Building wheels for collected packages: albumentations
      Building wheel for albumentations (setup.py) ... [?25l[?25hdone
      Created wheel for albumentations: filename=albumentations-0.4.6-py3-none-any.whl size=65174 sha256=28c520643eedf70531b685c5e83fd59f116f2177af1ee84c726b3b62639d6cf6
      Stored in directory: /root/.cache/pip/wheels/cf/34/0f/cb2a5f93561a181a4bcc84847ad6aaceea8b5a3127469616cc
    Successfully built albumentations
    Installing collected packages: imgaug, albumentations
      Attempting uninstall: imgaug
        Found existing installation: imgaug 0.2.9
        Uninstalling imgaug-0.2.9:
          Successfully uninstalled imgaug-0.2.9
      Attempting uninstall: albumentations
        Found existing installation: albumentations 0.1.12
        Uninstalling albumentations-0.1.12:
          Successfully uninstalled albumentations-0.1.12
    Successfully installed albumentations-0.4.6 imgaug-0.4.0



```python
# !pip list
# 두 버전 일치해야하고, 현재 4.1.2.30 버전 아니고 그 이후면 작동 albumentations 패키지 임포트 안됨
!pip list | grep opencv
```

    opencv-contrib-python         4.1.2.30
    opencv-python                 4.1.2.30


- config 파일 생성(pycharm에서는 파일 생성으로!)
  - colab이므로 따로 선언해둠


```python
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DATA_DIR = "/content/horse2zebra"
DATA_DIR = "/content/drive/MyDrive/pytorch-CycleGAN-and-pix2pix/datasets/horse2zebra"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMDA_IDENTITY = 0.0
LAMDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
     A.Resize(width=256,height=256),
     A.HorizontalFlip(p=0.5),
     # mean for all 3 channels
     A.Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value=255),
     ToTensorV2(),
    ],
    # horse : zebra
    # Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
    # the pipeline will augment those two images in the same way. 
    # it could be more than two, adding ", key:value" inside. 
    # 관련 parameter에 대한 참고내용 : https://albumentations.ai/docs/examples/example_multi_target/
    additional_targets={"image0": "image"}
)


```

  - utils file 생성(for saving and loading checkpoints)


```python
import random, torch, os, numpy as np
import torch.nn as nn
# import config
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        # state_dict()
        # state_dict 는 PyTorch에서 모델을 저장하거나 불러오는 데 관심이 있다면 필수적인 항목 
        # state_dict 객체는 Python 사전이기 때문에 쉽게 저장, 업데이트, 변경 및 복원할 수 있으며, 이는 PyTorch 모델과 옵티마이저에 엄청난 모듈성(modularity)을 제공
        # 이 때, 학습 가능한 매개변수를 갖는 계층(합성곱 계층, 선형 계층 등) 및 등록된 버퍼들(batchnorm의 running_mean)만 모델의 state_dict항목을 가진다는 점에 유의
        # 옵티마이저 객체( torch.optim ) 또한 옵티마이저의 상태 뿐만 아니라 사용된 하이퍼 매개변수 (Hyperparameter) 정보가 포함된 state_dict 을 갖습니다.(epoch 등등) 
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

  - Horse data set 로더 생성


```python
# import torch 
# to load images
from PIL import Image
import os
# # if you modularize all of these ipynb to python files later 
# import config
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
  def __init__(self, root_zebra, root_horse, transforms=None):
    self.root_zebra = root_zebra
    self.root_horse = root_horse
    self.transform = transforms

    # list all the image files inside the above root_zebra/_horse dir
    self.zebra_images = os.listdir(root_zebra)
    self.horse_images = os.listdir(root_horse)

    # data's are not paired, so the lenght of the data sets are not exactly equal
    # so we're checking what is the max length
    self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) #ex) 1000, 1500
    self.zebra_len = len(self.zebra_images)
    self.horse_len = len(self.horse_images)
  
  def __len__(self):
    #returning max len
    return self.length_dataset

  def __getitem__(self, index):

    # getting 'index'th image from the list of zebra images files
    # why modular %? index could be greater than the dataset that we have 
    # b/c we're taking the max out of two 
    # but, there could be flaws in that some images could be shown more often(b/c Dataset's arent paired)
    zebra_img = self.zebra_images[index % self.zebra_len]
    horse_img = self.horse_images[index % self.horse_len]
    # so far, this is just JPEG file in the root dir
    # to load img we have to have a PATH

    # joining img path with the root dir 
    zebra_path = os.path.join(self.root_zebra, zebra_img)
    horse_path = os.path.join(self.root_horse, horse_img)

    # converting those right above to PIL images or numpy array
    # numpy arrays in this case
    # our purpose here is to *alubumentate / Albumentations is a Python library for image augmentation.
    # so we're using numpy array of image.open
    # .convert("RGB") could actually be unnecessary but, just in case it's grayscale
    zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
    horse_img = np.array(Image.open(horse_path).convert("RGB"))

    if self.transform:
      # this way, we apply 'same' transformations to both of the images
      # So it will apply the same set of transformations with the same parameters.
      # b/c config 부분에서 additional_targets={"image0": "image"} 했기에 밑에 처럼 변형 적용
      augmentation = self.transform(image=zebra_img, image0 = horse_img)

      # image 다음 , image0이 같은 방식으로 augment되어 저장되어있음 augmentation(Dict) 안에
      # 고로 각각을 Key로 인덱스 접근해서 zebra_img, horse_img에 replace 
      zebra_img = augmentation["image"]
      horse_img = augmentation["image0"]

    return zebra_img, horse_img
```

  ## 학습 진행하기 



```python
import torch
# from datset import HorseZebraDataset
import sys
# from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import config
from tqdm import tqdm
from torchvision.utils import save_image
# from discriminator_model import Discriminator
# from generator_model import Generator

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
  # getting nice progress bar 
  loop = tqdm(loader, leave=True)
  
  # dataset 클래스에서 우리가 return 하는 게 zebra_img, horse_img니깐 (zebra, horse)로 loop돌린다
  for idx, (zebra, horse) in enumerate(loop):
    zebra = zebra.to(DEVICE)
    horse = horse.to(DEVICE)

    ###########################
    ####### TRAIN DISC ########
    ###########################

    #train disc_H, and disc_Z
    with torch.cuda.amp.autocast(): #float16일 때 필수과정임
      
      #####################
      ## TRAINING disc_H ##
      #####################
      fake_horse = gen_H(zebra)
      D_H_real = disc_H(horse) # disc on real horse
      # doing detach() b/c we'll use this fake_horse when we train gen_z, with detach() you don't have to retype in all these
      D_H_fake = disc_H(fake_horse.detach()) # disc on fake horse(generated horse)
      
      ###################################
      # disc_H의 loss function 정의 #
      # torch.ones_like(tensor) : tesor shape과 동일한 1로 채워진 matrix 생성 
      D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
      D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))

      # total D_H loss
      D_H_loss = D_H_real_loss + D_H_fake_loss

      #####################
      ## TRAINING disc_Z ##
      #####################
      fake_zebra = gen_Z(horse)
      D_Z_real = disc_Z(zebra) # disc result on real zebra
      # doing detach() b/c we'll use this fake_horse when we train gen_h, with detach() you don't have to retype in all these
      D_Z_fake = disc_Z(fake_zebra.detach()) # disc on fake zebra(generated zebra)
      
      ###################################
      # disc_Z의 loss function 정의 #
      # torch.ones_like(tensor) : tesor shape과 동일한 1로 채워진 matrix 생성(labeling) 
      D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
      D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))

      # total D_Z loss
      D_Z_loss = D_Z_real_loss + D_Z_fake_loss

      ## put it together ##
      D_loss = (D_H_loss + D_Z_loss) / 2 # as paper did
    
    # grad 초기화 후 
    opt_disc.zero_grad()
    # 역전파 수행해서 각 parameter 별 gradient구해서 저장해둠
    d_scaler.scale(D_loss).backward()
    d_scaler.step(opt_disc)
    d_scaler.update()





    ###########################
    ####### TRAIN GEN ########
    ###########################

    #train gen_H, and gen_Z
    with torch.cuda.amp.autocast(): #float16일 때 필수과정임
      
      ####################
      ## TRAINING gen_H ##
      ####################
      

      ## adversarial loss ##

      D_H_fake = disc_H(fake_horse) # disc result on fake horse
      # doing detach() b/c we'll use this fake_horse when we train gen_z, with detach() you don't have to retype in all these
      D_Z_fake = disc_Z(fake_zebra) # disc result on fake zebra
      
      G_H_loss = mse(D_H_fake, torch.ones_like(D_H_fake))
      G_Z_loss = mse(D_Z_fake, torch.ones_like(D_Z_fake))
      
      ## cycle loss ##
      
      cycle_zebra = gen_Z(fake_horse)
      cycle_horse = gen_H(fake_zebra)
      cycle_zebra_loss = L1(zebra, cycle_zebra)
      cycle_horse_loss = L1(horse, cycle_horse)

      ## identity loss ## (optional to preserve 'color preservation' when reconstructing input from the output)
      ## * 현재 LAMDA_IDENTITY = 0.0 인데, 추후에 쓰고 싶으면 상수만 변경해서 쓰면 된다. 

      # identity_zebra = gen_Z(zebra)
      # identity_horse = gen_H(horse)
      # identity_zebra_loss = L1(zebra, identity_zebra)
      # identity_horse_loss = L1(horse, identity_horse)

      ## put it together ##

      G_loss = (
          G_H_loss 
          + G_Z_loss 
          + cycle_zebra_loss * LAMDA_CYCLE
          + cycle_horse_loss * LAMDA_CYCLE
          # + identity_horse_loss * LAMDA_IDENTITY 
          # + identity_zebra_loss * LAMDA_IDENTITY  
      )


    # grad 초기화 후 
    opt_gen.zero_grad()
    # 역전파 수행해서 각 parameter 별 gradient구해서 저장해둠
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()

    if idx % 200 == 0:
      # why *0.5 + 0.5...?
      # inversing normalization we did in 'config' in A.Compose to augemnt the data
      fake_horse = fake_horse*0.5+0.5
      fake_zebra = fake_zebra*0.5+0.5
      result_horse = torch.cat((zebra.data, fake_horse.data), -2)
      result_zebra = torch.cat((horse.data, fake_zebra.data), -2)
      save_image(result_horse, f"saved_images/horse_{idx}.png")
      save_image(result_zebra, f"saved_images/zebra_{idx}.png")

  

def main():
  #initailizing D
  disc_H = Discriminator(in_channels=3).to(DEVICE) # discriminating horses
  disc_Z = Discriminator(in_channels=3).to(DEVICE) # discriminating zebras
  gen_H = Generator(img_channels=3, num_residuals=9).to(DEVICE) # generating horses
  gen_Z = Generator(img_channels=3, num_residuals=9).to(DEVICE) # generating zebras
  
  opt_disc = optim.Adam(
      # parameters(): Returns an iterator over module parameters. This is typically passed to an optimizer.
      # betas : coefficients used for computing running averages of gradient and its square 
      # '+' each list : concatenating
      # why concat? to minimize the number of optimizer(4->2)
      list(disc_H.parameters()) + list(disc_Z.parameters()),
      lr = LEARNING_RATE,
      # 0.5 for momentum term / 0.999 for beta2
      betas = (0.5,0.999)
  )
  
  opt_gen = optim.Adam(
      list(gen_Z.parameters()) + list(gen_H.parameters()),
      lr = LEARNING_RATE,
      betas = (0.5,0.999)
  )

  # for cycle-consistency loss / identity loss
  L1 = nn.L1Loss()
  # advaersarial loss(=GAN loss)
  mse = nn.MSELoss()

  if LOAD_MODEL:
    load_checkpoint(
        CHECKPOINT_GEN_H, gen_H, opt_gen, LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_GEN_Z, gen_Z, opt_gen, LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_CRITIC_H, disc_H, opt_disc, LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, LEARNING_RATE,
    )


  dataset = HorseZebraDataset(
      # trainA : horse / trainB : zebra 
      # 영상과는 약간 다르게 코딩함, original paper의 dataset으로 진행
      root_horse=DATA_DIR+"/trainA", root_zebra=DATA_DIR+"/trainB", transforms=transforms
  )

  loader = DataLoader(
      dataset,
      batch_size = BATCH_SIZE,
      shuffle = True,
      num_workers=NUM_WORKERS,
      pin_memory=True
  )

  # float16으로 진행한다면 실행 / float32으로 진행한다면 실행 안해도됨
  # always nice to run in float16
  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()


  for epoch in range(NUM_EPOCHS):
    train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

    if SAVE_MODEL:
      save_checkpoint(gen_H, opt_gen, filename=CHECKPOINT_GEN_H)
      save_checkpoint(gen_Z, opt_gen, filename=CHECKPOINT_GEN_Z)
      save_checkpoint(disc_H, opt_disc, filename=CHECKPOINT_CRITIC_H)
      save_checkpoint(disc_Z, opt_disc, filename=CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
  main()

 



```

    => Loading checkpoint
    => Loading checkpoint
    => Loading checkpoint
    => Loading checkpoint


    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    100%|██████████| 1334/1334 [06:38<00:00,  3.34it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [03:58<00:00,  5.58it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [03:59<00:00,  5.57it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [03:58<00:00,  5.59it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [03:58<00:00,  5.59it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [03:58<00:00,  5.59it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:01<00:00,  5.53it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.54it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.54it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.55it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


    100%|██████████| 1334/1334 [04:00<00:00,  5.54it/s]


    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint
    => Saving checkpoint


     24%|██▎       | 314/1334 [00:57<03:06,  5.47it/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-4017041713b1> in <module>()
        218 
        219 if __name__ == "__main__":
    --> 220   main()
        221 
        222 


    <ipython-input-9-4017041713b1> in main()
        208 
        209   for epoch in range(NUM_EPOCHS):
    --> 210     train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        211 
        212     if SAVE_MODEL:


    <ipython-input-9-4017041713b1> in train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
         99       ## cycle loss ##
        100 
    --> 101       cycle_zebra = gen_Z(fake_horse)
        102       cycle_horse = gen_H(fake_zebra)
        103       cycle_zebra_loss = L1(zebra, cycle_zebra)


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    <ipython-input-2-4fe02f98518c> in forward(self, x)
         86       x = layer(x)
         87 
    ---> 88     x = self.residual_blocks(x)
         89 
         90     for layer in self.up_blocks:


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    <ipython-input-2-4fe02f98518c> in forward(self, x)
         31   def forward(self, x):
         32     # concatenating previous low-level info
    ---> 33     return  x + self.block(x)
         34 
         35 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    <ipython-input-2-4fe02f98518c> in forward(self, x)
         15 
         16   def forward(self, x):
    ---> 17     return self.conv(x)
         18 
         19 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/instancenorm.py in forward(self, input)
         57         return F.instance_norm(
         58             input, self.running_mean, self.running_var, self.weight, self.bias,
    ---> 59             self.training or not self.track_running_stats, self.momentum, self.eps)
         60 
         61 


    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py in instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
       2326         _verify_spatial_size(input.size())
       2327     return torch.instance_norm(
    -> 2328         input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch.backends.cudnn.enabled
       2329     )
       2330 


    KeyboardInterrupt: 


## 결과 출력
    - Zebra -> Horse

![horse_400.png](/assets/horse_400.png)

![horse_600.png](/assets/horse_600.png)

![horse_1200.png](/assets/horse_1200.png)

![horse_1000.png](/assets/horse_1000.png)

![horse_1200%20%281%29.png](/assets/horse_1200%20%281%29.png)

    - Horse -> Zebra

![zebra_400.png](/assets/zebra_400.png)

![zebra_800%20%281%29.png](/assets/zebra_800%20%281%29.png)

![zebra_800.png](/assets/zebra_800.png)

![zebra_1200%20%281%29.png](/assets/zebra_1200%20%281%29.png)

![zebra_1200.png](/assets/zebra_1200.png)

  ## 테스트 데이터 출력해보기

  - 테스트 HORSE IMAGE 출력


```python
from PIL import Image
import matplotlib.pyplot as plt
# for i in range (10):
#   imgs = next(iter(test_dataloader)) 
#   real_horse = imgs[]

image = Image.open('/content/drive/MyDrive/pytorch-CycleGAN-and-pix2pix/datasets/horse2zebra/testA/n02381460_1210.jpg')
print("image size: ", image.size)
plt.imshow(image)
plt.show()
```

    image size:  (256, 256)



    
![png](/assets/output_40_1.png)
    


  - 학습한 모델 불러와서 IMAGE TRANSLATION 적용해보기


```python
# # not done yet # # 

# def generate_images(model, test_input):
#   prediction = model(test_input)

#   plt.figure(figsize=(12, 12))

#   display_list = [test_input[0], prediction[0]]
#   title = ['Input Image', 'Predicted Image']

#   for i in range(2):
#     plt.subplot(1, 2, i+1)
#     plt.title(title[i])
#     # getting the pixel values between [0, 1] to plot it.
#     plt.imshow(display_list[i] * 0.5 + 0.5)
#     plt.axis('off')
#   plt.show()


# test_dataset = HorseZebraDataset(
#     root_horse=DATA_DIR+"/testA", root_zebra=DATA_DIR+"/testB", transforms=transforms
#     )

# test_dataloader = DataLoader(
#     test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
#     )

# zebra, horse = test_dataset.__getitem__(0)

# # PIL -> np.array = np.array( ) 또는 np.asarray( ) 함수를 이용하여 PIL Image를 NumPy array로 변환할 수 있다.
# # np.array -> PIL = PIL 패키지의 Image.fromarray() 를 사용하면 된다.
# # tesor(pytorch) -> PIL = from torchvision.transforms.functional import to_pil_image / to_pil_image 


# # model initialize
# gen_H = Generator(img_channels=3, num_residuals=9).to(DEVICE) # generating horses
# gen_Z = Generator(img_channels=3, num_residuals=9).to(DEVICE) # generating zebras

# # model.load_state_dict(torch.load('model_weights.pth')

# checkpoint_gen_h = torch.load(CHECKPOINT_GEN_H)
# checkpoint_gen_z = torch.load(CHECKPOINT_GEN_Z)
# gen_H.load_state_dict(checkpoint_gen_h["state_dict"])
# gen_Z.load_state_dict(checkpoint_gen_z["state_dict"])

# gen_H.eval()
# gen_Z.eval()

# # CHECKPOINT_GEN_H = "genh.pth.tar"
# # CHECKPOINT_GEN_Z = "genz.pth.tar"
# # CHECKPOINT_CRITIC_H = "critich.pth.tar"
# # CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

# fake_horse = gen_H(zebra)
# fake_zebra = gen_Z(horse)

# # Run the trained model on the test dataset
# for inp in test_horses.take(5):
#   generate_images(generator_g, inp)
```

    (256, 256)


    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))



    
![png](/assets/output_42_2.png)
    



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-73-6c2eb7e2fa33> in <module>()
         45 # CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
         46 
    ---> 47 fake_horse = gen_H(zebra)
         48 fake_zebra = gen_Z(horse)
         49 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    <ipython-input-2-4fe02f98518c> in forward(self, x)
         81   def forward(self, x):
         82 
    ---> 83     x = self.initial(x)
         84 
         85     for layer in self.down_blocks:


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/conv.py in forward(self, input)
        444 
        445     def forward(self, input: Tensor) -> Tensor:
    --> 446         return self._conv_forward(input, self.weight, self.bias)
        447 
        448 class Conv3d(_ConvNd):


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
        439             return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
        440                             weight, bias, self.stride,
    --> 441                             _pair(0), self.dilation, self.groups)
        442         return F.conv2d(input, weight, bias, self.stride,
        443                         self.padding, self.dilation, self.groups)


    RuntimeError: Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 3-dimensional input of size [3, 262, 262] instead

