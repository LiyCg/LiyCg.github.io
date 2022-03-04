# GAN

2014년 Ian Goodfellow가 발표한 논문이다. 

오늘날에도 활발히 연구되어 다양한 AI application에 활용되고 있는 딥러닝 생성모델로 컴퓨터 비전쪽에서는 혁신이었다. 

네이버 블로그에서 리뷰한 카이스트 VML 연구실의 딥러닝 기반의 리타게팅 시스템에 관한 논문을 읽고 VAE를 공부했다. 

이 매우 신기한 '생성 모델'에 대해 맛을 보고 나니, 옆에서 더 맛있어보이는 GAN이 떡하니 있었다는 걸 알게 되었다. 

GAN 논문을 읽어보고 실습을 통해 내가 만든 모델이 이미지를 '생성'해내는 것을 보니, 과장되게 말하면 내가 만든 생명체가 '그림'을 그리는 것 같은 느낌이었다. 

긴말 말고 실습을 시작해보자. 


## 1) 특징

- The Start of focus on creating new sample itself 

- VAE explicitly approximates the distribution of original data

- GAN does it implicitly

- To simplyfy the practice, I'll use MNIST(1X28X28) data set




```python
# load dataset
import torch
import torch.nn as nn

from torchvision import datasets
# data 전처리를 위해 transforms 불러옴
import torchvision.transforms as transforms
from torchvision.utils import save_image

# defining Generator and Discriminator
latent_dim = 100

# definition of GENERATOR class
# Generator class가 nn.Module class에서 상속받음
class Generator(nn.Module):
  
  def __init__(self):
    
    # super(): 자식클래서에서 부모클래스의 func을 사용하고 싶은경우
    # 즉, 기본적으로 Generator class의 __init__()이 nn.Module의 __init() 을 그대로 상속받도록..!
    # This is useful for accessing inherited methods that have been overridden in a class.
    # super().method(arg) = super(curClass, self).method(arg)
    # method(arg) = override or inherit하려는 부모클래스의 func
    super(Generator, self).__init__()
    

    # defining one block
    def block(input_dim, output_dim, normalize=True):
      
      # nn.Linear => Linear Regressor with specifying input and output dims
      # it's a list
      layers = [nn.Linear(input_dim, output_dim)]

      if normalize:
        # batch normalization on(between same dim)
        # BN 하는 이유?
        # - 배치별로 학습하게 되면 Internal Covariant Shift가 발생(배치별로 데이터 분포가 다르다는 것)
        # - 이 문제를 해결위해 표준정규분포(평균 0, 편차 1)로 shift해서 동일하게 맞춰주는 것(=a.k.a 정규화)!
        # what is 0.8? 
        # - 추후 ReLU에 의해 음수 부분이 전부 0이 되지않도록 정규화 된 값에 더해져서 음수부의 활성화값이 모두 0이되는 것 완화
        layers.append(nn.BatchNorm1d(output_dim,0.8))

      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers 
  
    # Generator posesses a number of continuous blocks
    # nn.Sequential() is normally used to represent Neural Network
    # nn.Sequential()을 사용하면 __init__()에서 사용할 네트워크 모델들을 정의해줌
    #                          순전파를 layer 형태로 가독성이 뛰어나게 코드 작성할 수 있음
    # NN가 deep해질수록 효과가 높음
    # why '*'? to access the element of layers 
    self.model = nn.Sequential(
        # dimension이 점점 확장되는 것이 보일 것
        *block(latent_dim, 128, normalize=True),
        *block(128,256),
        *block(256,512),
        *block(512, 1024),
        # final output
        nn.Linear(1024, 1*28*28),
        # -1에서 1사이 값 같도록!
        nn.Tanh() #activation Func
    )
  # __init__() 정의 끝


  def forward(self, z):
    img = self.model(z)
    #.view(*shape) -> returns a new tensor with the same data the 'self' tensor, but different 'shape'
    # - basically it means to copy and paste the data in image and change it's shape to 'shape'
    # img.size(0) = batch size
    # img의 형태로 만듬 
    img = img.view(img.size(0), 1, 28, 28)
    return img

# GENERATOR class 정의 끝


##################################################################

# definition of DISCRIMINATOR class
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(1*28*28, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256,1),
        # sigmoid()로 확률값을 내보내도록!
        nn.Sigmoid()
    )

  # 이미지에 대한 판별 결과 반환(Regression 즉, real일 Probability 구하기)
  def forward(self, img):
    # data.size(mini-batch size, channel size, img height, img width)
    # -1 refers to last dimension(in this case 28, which is width because it's 1X28X28)
    # 즉, batch 전체를 직렬화
    flattened = img.view(img.size(0), -1)
    output = self.model(flattened) 

    return output

# DISCRIMINATOR class 정의 끝


##################################################################


```

## 2) Loading and importing MNIST data set


```python
# loading MNIST
transforms_train = transforms.Compose([
  # resize to 28x28                                     
  transforms.Resize(28),
  # pytorch의 tensor형태로 사용할 수 있도록 만듬
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])   
])

# train = True로 학습데이터만 불러와서 정의한 transforms_train 전처리 함수로 전처리한다!
train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)

# Dataset 은 데이터셋의 특징(feature)을 가져오고 하나의 샘플에 정답(label)을 지정하는 일을 한 번에 합니다. 
# 모델을 학습할 때, 일반적으로 샘플들을 “미니배치(minibatch)”로 전달하고, 
# 매 에폭(epoch)마다 데이터를 다시 섞어서 과적합(overfit)을 막고, 
# Python의 multiprocessing 을 사용하여 데이터 검색 속도를 높이려고 합니다.
# DataLoader 는 간단한 API로 이러한 복잡한 과정들을 추상화한 순회 가능한 객체(iterable)입니다.

# batch 하나가 128개가 될 수 있도록 만듬
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)


```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/train-images-idx3-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/train-labels-idx1-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/t10k-images-idx3-ubyte.gz to ./dataset/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./dataset/MNIST/raw
    


## 3) GAN 모델학습 & 샘플링

- 학습 위해 G, D를 초기화

- 적절한 Hyperparameter 설정


```python
# 모델학습 및 샘플링

#initialization
generator = Generator()
discriminator = Discriminator()

# Moves all model parameters and buffers to the GPU. 즉, GPU로 올리는 것!
# This also makes associated parameters and buffers different objects. 
# So it should be called before constructing optimizer if the module will live on GPU while being optimized.
generator.cuda()
discriminator.cuda()

# loss function
# Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# 참고로, lr과 betas는 일반적으로 가장 많이 활용되는 hyperparmeter를 그대로 이용했다
# learning rate set up
lr = 0.0002

# setting up G, D's optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))

```

## 4) 모델을 학습하면서 peoridic 하게 sampling하며 결과 확인 가능하게 하기


```python
import time

n_epochs = 200 #학습의 횟수(epoch) 설정
sample_interval = 2000 # 몇번의 배치(batch)마다 결과 출력 할 것인지 설정
start_time = time.time()

# 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능을 가집니다.
# enumerate는 “열거하다”라는 뜻입니다.
# 이 함수는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴합니다.
# 보통 enumerate 함수는 for문과 함께 자주 사용됩니다.
for epoch in range(n_epochs):
  for i, (imgs, _) in enumerate(dataloader):

    # real 이미지와 fake 이미지에 대한 labeling
    # img.size(0) 즉, batch size만큼의 진짜 label과 가짜 label을 만듬
    real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # (real) :1
    fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # (fake) :0

    real_imgs = imgs.cuda()


    """ Training G """
    optimizer_G.zero_grad()
    
    # random noise(latent vector) sampling
    z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()

    # image 생성
    generated_imgs = generator(z)

    # G의 Loss func 계산
    # G 입장에서는 generated_imgs 즉, G(z)가 real로 구분될 수 있도록 해야함
    # 즉, discriminator(generated_imgs)가 real이 될수있도록, 그러한 방향으로 손실함수 정의!
    g_loss = adversarial_loss(discriminator(generated_imgs), real)

    # updating G first
    g_loss.backward()
    optimizer_G.step()


    """ Training D """
    # 한 루프에서 업데이트를 위해 loss.backward()를 호출하면 각 파라미터들의 .grad 값에 변화도가 저장이 된다.
    # 이후 다음 루프에서 zero_grad()를 하지않고 역전파를 시키면 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭을 해서 원하는 방향으로 학습이 안된다고 한다.
    # 따라서 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화시킨 후 학습을 진행해야 한다.
    optimizer_D.zero_grad()

    # D의 Loss Value 계산
    # D 입장에서는 real_imgs 즉, x가 real로 구분될 수 있도록 해야함
    # 즉, discriminator(real_imgs)가 real이 될수있도록, 그러한 방향으로 손실함수 정의!
    real_loss = adversarial_loss(discriminator(real_imgs), real)

    # detach(): 기존 Tensor에서 gradient 전파가 안되는 텐서 생성
    # 단 storage를 공유하기에 detach로 생성한 Tensor가 변경되면 원본 Tensor도 똑같이 변합니다.
    # 반면, generated_imgs는 즉, G(z)가 fake로 구분될 수 있도록 해야함
    # 즉, discriminator(generated_imgs)가 fake가 될수있도록, 그러한 방향으로 손실함수 정의!
    fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    # updating D next 
    # GAN 논문과 달리 학습순서가 G->D 임, 그러나 마찬가지로 학습이 잘됨
    d_loss.backward()
    optimizer_D.step()

    done = epoch * len(dataloader) + i
    if done % sample_interval == 0:
      # generated_imgs 중 25개 선택해서 5X5 격자 이미지에 출력
      save_image(generated_imgs.data[:25], f"{done}.png", nrow = 5, normalize=True)

  # epoch 끝날 때마다 로그(log) 출력
  print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss:: {g_loss.item():.6f}] [Elapsed Time: {time.time() - start_time:.2f}s")













```

    [Epoch 0/200] [D loss: 0.682269] [G loss:: 0.351552] [Elapsed Time: 19.42s
    [Epoch 1/200] [D loss: 0.561312] [G loss:: 2.071228] [Elapsed Time: 38.81s
    [Epoch 2/200] [D loss: 0.327124] [G loss:: 1.379791] [Elapsed Time: 58.31s
    [Epoch 3/200] [D loss: 0.247093] [G loss:: 2.518109] [Elapsed Time: 78.11s
    [Epoch 4/200] [D loss: 0.182192] [G loss:: 1.867597] [Elapsed Time: 98.10s
    [Epoch 5/200] [D loss: 0.395076] [G loss:: 0.875088] [Elapsed Time: 118.01s
    [Epoch 6/200] [D loss: 0.239915] [G loss:: 1.948697] [Elapsed Time: 137.79s
    [Epoch 7/200] [D loss: 0.226691] [G loss:: 2.058844] [Elapsed Time: 157.62s
    [Epoch 8/200] [D loss: 0.254351] [G loss:: 2.540133] [Elapsed Time: 177.57s
    [Epoch 9/200] [D loss: 0.233253] [G loss:: 2.184985] [Elapsed Time: 197.38s
    [Epoch 10/200] [D loss: 0.222781] [G loss:: 2.160261] [Elapsed Time: 217.17s
    [Epoch 11/200] [D loss: 0.194285] [G loss:: 2.798955] [Elapsed Time: 237.02s
    [Epoch 12/200] [D loss: 0.180951] [G loss:: 2.971715] [Elapsed Time: 256.72s
    [Epoch 13/200] [D loss: 0.235931] [G loss:: 4.233338] [Elapsed Time: 276.78s
    [Epoch 14/200] [D loss: 0.549759] [G loss:: 5.144494] [Elapsed Time: 296.94s
    [Epoch 15/200] [D loss: 0.139545] [G loss:: 2.359173] [Elapsed Time: 316.92s
    [Epoch 16/200] [D loss: 0.126066] [G loss:: 2.194533] [Elapsed Time: 337.03s
    [Epoch 17/200] [D loss: 0.179281] [G loss:: 2.954103] [Elapsed Time: 356.99s
    [Epoch 18/200] [D loss: 0.143988] [G loss:: 2.307570] [Elapsed Time: 377.00s
    [Epoch 19/200] [D loss: 0.742042] [G loss:: 7.275518] [Elapsed Time: 397.08s
    [Epoch 20/200] [D loss: 0.078452] [G loss:: 2.389386] [Elapsed Time: 417.04s
    [Epoch 21/200] [D loss: 0.144276] [G loss:: 2.281540] [Elapsed Time: 437.05s
    [Epoch 22/200] [D loss: 0.264275] [G loss:: 1.082899] [Elapsed Time: 456.96s
    [Epoch 23/200] [D loss: 0.118844] [G loss:: 2.171868] [Elapsed Time: 476.82s
    [Epoch 24/200] [D loss: 0.230996] [G loss:: 2.345797] [Elapsed Time: 496.85s
    [Epoch 25/200] [D loss: 0.225230] [G loss:: 2.608441] [Elapsed Time: 516.80s
    [Epoch 26/200] [D loss: 0.140656] [G loss:: 2.450466] [Elapsed Time: 536.74s
    [Epoch 27/200] [D loss: 0.146317] [G loss:: 3.078031] [Elapsed Time: 556.60s
    [Epoch 28/200] [D loss: 0.157054] [G loss:: 3.810877] [Elapsed Time: 576.40s
    [Epoch 29/200] [D loss: 0.118798] [G loss:: 2.779831] [Elapsed Time: 596.33s
    [Epoch 30/200] [D loss: 0.154727] [G loss:: 3.188426] [Elapsed Time: 616.14s
    [Epoch 31/200] [D loss: 0.257376] [G loss:: 1.256821] [Elapsed Time: 636.16s
    [Epoch 32/200] [D loss: 0.162950] [G loss:: 2.623882] [Elapsed Time: 656.07s
    [Epoch 33/200] [D loss: 1.596372] [G loss:: 9.497270] [Elapsed Time: 676.25s
    [Epoch 34/200] [D loss: 0.153785] [G loss:: 2.104753] [Elapsed Time: 695.97s
    [Epoch 35/200] [D loss: 0.131418] [G loss:: 4.737673] [Elapsed Time: 715.64s
    [Epoch 36/200] [D loss: 0.180871] [G loss:: 6.362998] [Elapsed Time: 735.61s
    [Epoch 37/200] [D loss: 0.139901] [G loss:: 2.224628] [Elapsed Time: 755.52s
    [Epoch 38/200] [D loss: 0.149496] [G loss:: 2.992026] [Elapsed Time: 775.19s
    [Epoch 39/200] [D loss: 0.278004] [G loss:: 1.778815] [Elapsed Time: 795.10s
    [Epoch 40/200] [D loss: 0.070008] [G loss:: 3.047331] [Elapsed Time: 815.03s
    [Epoch 41/200] [D loss: 0.267727] [G loss:: 1.361007] [Elapsed Time: 834.78s
    [Epoch 42/200] [D loss: 0.346834] [G loss:: 8.005226] [Elapsed Time: 854.32s
    [Epoch 43/200] [D loss: 0.263824] [G loss:: 1.585345] [Elapsed Time: 874.16s
    [Epoch 44/200] [D loss: 0.139149] [G loss:: 4.519037] [Elapsed Time: 893.80s
    [Epoch 45/200] [D loss: 0.151465] [G loss:: 2.406449] [Elapsed Time: 913.56s
    [Epoch 46/200] [D loss: 0.093131] [G loss:: 2.920806] [Elapsed Time: 933.37s
    [Epoch 47/200] [D loss: 0.141780] [G loss:: 2.912453] [Elapsed Time: 952.95s
    [Epoch 48/200] [D loss: 0.174404] [G loss:: 1.927529] [Elapsed Time: 972.88s
    [Epoch 49/200] [D loss: 0.154456] [G loss:: 1.751621] [Elapsed Time: 992.46s
    [Epoch 50/200] [D loss: 0.166301] [G loss:: 2.422437] [Elapsed Time: 1011.89s
    [Epoch 51/200] [D loss: 0.180479] [G loss:: 3.325277] [Elapsed Time: 1031.51s
    [Epoch 52/200] [D loss: 0.180604] [G loss:: 5.307438] [Elapsed Time: 1051.23s
    [Epoch 53/200] [D loss: 0.180217] [G loss:: 2.393197] [Elapsed Time: 1071.15s
    [Epoch 54/200] [D loss: 0.198832] [G loss:: 3.193680] [Elapsed Time: 1090.81s
    [Epoch 55/200] [D loss: 0.161580] [G loss:: 2.868010] [Elapsed Time: 1110.61s
    [Epoch 56/200] [D loss: 0.736643] [G loss:: 12.816195] [Elapsed Time: 1130.54s
    [Epoch 57/200] [D loss: 0.143047] [G loss:: 2.552904] [Elapsed Time: 1150.59s
    [Epoch 58/200] [D loss: 0.094474] [G loss:: 5.047415] [Elapsed Time: 1170.57s
    [Epoch 59/200] [D loss: 0.137185] [G loss:: 3.145074] [Elapsed Time: 1190.38s
    [Epoch 60/200] [D loss: 0.155445] [G loss:: 3.141008] [Elapsed Time: 1210.53s
    [Epoch 61/200] [D loss: 0.385133] [G loss:: 0.847569] [Elapsed Time: 1230.42s
    [Epoch 62/200] [D loss: 0.122449] [G loss:: 2.704200] [Elapsed Time: 1250.15s
    [Epoch 63/200] [D loss: 0.101272] [G loss:: 2.331720] [Elapsed Time: 1270.07s
    [Epoch 64/200] [D loss: 0.146732] [G loss:: 4.492449] [Elapsed Time: 1289.88s
    [Epoch 65/200] [D loss: 0.163871] [G loss:: 2.995676] [Elapsed Time: 1309.89s
    [Epoch 66/200] [D loss: 0.170727] [G loss:: 2.086760] [Elapsed Time: 1329.88s
    [Epoch 67/200] [D loss: 0.215617] [G loss:: 4.674460] [Elapsed Time: 1349.87s
    [Epoch 68/200] [D loss: 0.209865] [G loss:: 2.231739] [Elapsed Time: 1369.78s
    [Epoch 69/200] [D loss: 0.147198] [G loss:: 2.344101] [Elapsed Time: 1389.83s
    [Epoch 70/200] [D loss: 0.128067] [G loss:: 2.509690] [Elapsed Time: 1409.81s
    [Epoch 71/200] [D loss: 0.747902] [G loss:: 9.033361] [Elapsed Time: 1429.65s
    [Epoch 72/200] [D loss: 0.115800] [G loss:: 2.460217] [Elapsed Time: 1449.61s
    [Epoch 73/200] [D loss: 0.493707] [G loss:: 1.162146] [Elapsed Time: 1469.51s
    [Epoch 74/200] [D loss: 0.125383] [G loss:: 3.103346] [Elapsed Time: 1489.49s
    [Epoch 75/200] [D loss: 0.143661] [G loss:: 3.189035] [Elapsed Time: 1509.27s
    [Epoch 76/200] [D loss: 0.191987] [G loss:: 2.143736] [Elapsed Time: 1528.97s
    [Epoch 77/200] [D loss: 0.135503] [G loss:: 4.025548] [Elapsed Time: 1548.77s
    [Epoch 78/200] [D loss: 0.137421] [G loss:: 3.224050] [Elapsed Time: 1568.61s
    [Epoch 79/200] [D loss: 0.163558] [G loss:: 3.185355] [Elapsed Time: 1588.48s
    [Epoch 80/200] [D loss: 0.199294] [G loss:: 3.192581] [Elapsed Time: 1608.19s
    [Epoch 81/200] [D loss: 0.165450] [G loss:: 2.855398] [Elapsed Time: 1627.95s
    [Epoch 82/200] [D loss: 0.093801] [G loss:: 3.927411] [Elapsed Time: 1647.66s
    [Epoch 83/200] [D loss: 0.139092] [G loss:: 2.963875] [Elapsed Time: 1667.35s
    [Epoch 84/200] [D loss: 0.114821] [G loss:: 4.717634] [Elapsed Time: 1686.83s
    [Epoch 85/200] [D loss: 0.108526] [G loss:: 2.479369] [Elapsed Time: 1706.57s
    [Epoch 86/200] [D loss: 0.079528] [G loss:: 3.078678] [Elapsed Time: 1726.26s
    [Epoch 87/200] [D loss: 0.235793] [G loss:: 2.438574] [Elapsed Time: 1745.84s
    [Epoch 88/200] [D loss: 0.201118] [G loss:: 3.926283] [Elapsed Time: 1765.48s
    [Epoch 89/200] [D loss: 0.129247] [G loss:: 3.543575] [Elapsed Time: 1785.51s
    [Epoch 90/200] [D loss: 0.127121] [G loss:: 3.685171] [Elapsed Time: 1805.27s
    [Epoch 91/200] [D loss: 0.040292] [G loss:: 4.335946] [Elapsed Time: 1825.10s
    [Epoch 92/200] [D loss: 0.268462] [G loss:: 2.234693] [Elapsed Time: 1844.94s
    [Epoch 93/200] [D loss: 0.085448] [G loss:: 3.432781] [Elapsed Time: 1864.69s
    [Epoch 94/200] [D loss: 0.174905] [G loss:: 4.267883] [Elapsed Time: 1884.47s
    [Epoch 95/200] [D loss: 0.796720] [G loss:: 0.563536] [Elapsed Time: 1904.40s
    [Epoch 96/200] [D loss: 0.137758] [G loss:: 3.204298] [Elapsed Time: 1924.07s
    [Epoch 97/200] [D loss: 0.141559] [G loss:: 3.731245] [Elapsed Time: 1943.86s
    [Epoch 98/200] [D loss: 0.117802] [G loss:: 2.258639] [Elapsed Time: 1963.73s
    [Epoch 99/200] [D loss: 0.178104] [G loss:: 1.844639] [Elapsed Time: 1983.24s
    [Epoch 100/200] [D loss: 0.235183] [G loss:: 2.585296] [Elapsed Time: 2003.07s
    [Epoch 101/200] [D loss: 0.150372] [G loss:: 2.679521] [Elapsed Time: 2022.80s
    [Epoch 102/200] [D loss: 0.149943] [G loss:: 4.496220] [Elapsed Time: 2042.68s
    [Epoch 103/200] [D loss: 0.132484] [G loss:: 2.448923] [Elapsed Time: 2062.43s
    [Epoch 104/200] [D loss: 0.179656] [G loss:: 2.685145] [Elapsed Time: 2082.07s
    [Epoch 105/200] [D loss: 0.106539] [G loss:: 4.275385] [Elapsed Time: 2101.89s
    [Epoch 106/200] [D loss: 0.098557] [G loss:: 3.204100] [Elapsed Time: 2121.76s
    [Epoch 107/200] [D loss: 0.409181] [G loss:: 8.335756] [Elapsed Time: 2141.57s
    [Epoch 108/200] [D loss: 0.059060] [G loss:: 3.826717] [Elapsed Time: 2161.38s
    [Epoch 109/200] [D loss: 0.134694] [G loss:: 2.824901] [Elapsed Time: 2181.22s
    [Epoch 110/200] [D loss: 0.061382] [G loss:: 3.155811] [Elapsed Time: 2201.14s
    [Epoch 111/200] [D loss: 0.117363] [G loss:: 3.266145] [Elapsed Time: 2220.98s
    [Epoch 112/200] [D loss: 0.145440] [G loss:: 6.286451] [Elapsed Time: 2240.67s
    [Epoch 113/200] [D loss: 0.060121] [G loss:: 4.581471] [Elapsed Time: 2260.65s
    [Epoch 114/200] [D loss: 0.112771] [G loss:: 3.795421] [Elapsed Time: 2280.44s
    [Epoch 115/200] [D loss: 0.102281] [G loss:: 2.865645] [Elapsed Time: 2300.26s
    [Epoch 116/200] [D loss: 0.110525] [G loss:: 3.645791] [Elapsed Time: 2320.23s
    [Epoch 117/200] [D loss: 0.166755] [G loss:: 4.654974] [Elapsed Time: 2340.15s
    [Epoch 118/200] [D loss: 0.125399] [G loss:: 2.341789] [Elapsed Time: 2360.11s
    [Epoch 119/200] [D loss: 0.126917] [G loss:: 3.176652] [Elapsed Time: 2380.01s
    [Epoch 120/200] [D loss: 0.132827] [G loss:: 1.882506] [Elapsed Time: 2400.05s
    [Epoch 121/200] [D loss: 0.031942] [G loss:: 3.830306] [Elapsed Time: 2419.84s
    [Epoch 122/200] [D loss: 0.187298] [G loss:: 3.446159] [Elapsed Time: 2439.70s
    [Epoch 123/200] [D loss: 0.085885] [G loss:: 3.114878] [Elapsed Time: 2459.50s
    [Epoch 124/200] [D loss: 0.110789] [G loss:: 3.206274] [Elapsed Time: 2479.62s
    [Epoch 125/200] [D loss: 0.553504] [G loss:: 8.299034] [Elapsed Time: 2499.37s
    [Epoch 126/200] [D loss: 0.143216] [G loss:: 2.930615] [Elapsed Time: 2519.30s
    [Epoch 127/200] [D loss: 0.131474] [G loss:: 4.221880] [Elapsed Time: 2539.25s
    [Epoch 128/200] [D loss: 0.264724] [G loss:: 3.733158] [Elapsed Time: 2559.09s
    [Epoch 129/200] [D loss: 0.107586] [G loss:: 4.208236] [Elapsed Time: 2578.93s
    [Epoch 130/200] [D loss: 0.283192] [G loss:: 2.448572] [Elapsed Time: 2599.28s
    [Epoch 131/200] [D loss: 0.081678] [G loss:: 3.201787] [Elapsed Time: 2619.36s
    [Epoch 132/200] [D loss: 0.151465] [G loss:: 4.003814] [Elapsed Time: 2639.31s
    [Epoch 133/200] [D loss: 0.128588] [G loss:: 2.097712] [Elapsed Time: 2659.24s
    [Epoch 134/200] [D loss: 0.154764] [G loss:: 2.924391] [Elapsed Time: 2679.28s
    [Epoch 135/200] [D loss: 0.150435] [G loss:: 3.151100] [Elapsed Time: 2699.31s
    [Epoch 136/200] [D loss: 0.225610] [G loss:: 4.365973] [Elapsed Time: 2719.06s
    [Epoch 137/200] [D loss: 0.440759] [G loss:: 12.813972] [Elapsed Time: 2738.94s
    [Epoch 138/200] [D loss: 0.077490] [G loss:: 2.999319] [Elapsed Time: 2758.73s
    [Epoch 139/200] [D loss: 0.126187] [G loss:: 5.267907] [Elapsed Time: 2778.69s
    [Epoch 140/200] [D loss: 0.192952] [G loss:: 1.464762] [Elapsed Time: 2798.73s
    [Epoch 141/200] [D loss: 0.087916] [G loss:: 3.234757] [Elapsed Time: 2818.67s
    [Epoch 142/200] [D loss: 0.122765] [G loss:: 3.679382] [Elapsed Time: 2838.94s
    [Epoch 143/200] [D loss: 0.138357] [G loss:: 5.258627] [Elapsed Time: 2859.14s
    [Epoch 144/200] [D loss: 0.125558] [G loss:: 2.955918] [Elapsed Time: 2879.24s
    [Epoch 145/200] [D loss: 0.106319] [G loss:: 3.248249] [Elapsed Time: 2899.38s
    [Epoch 146/200] [D loss: 0.117072] [G loss:: 4.364952] [Elapsed Time: 2919.48s
    [Epoch 147/200] [D loss: 0.259479] [G loss:: 1.697480] [Elapsed Time: 2939.55s
    [Epoch 148/200] [D loss: 0.176206] [G loss:: 5.229673] [Elapsed Time: 2959.69s
    [Epoch 149/200] [D loss: 0.144065] [G loss:: 3.567263] [Elapsed Time: 2979.87s
    [Epoch 150/200] [D loss: 0.175993] [G loss:: 2.004480] [Elapsed Time: 2999.70s
    [Epoch 151/200] [D loss: 0.136987] [G loss:: 3.244904] [Elapsed Time: 3019.78s
    [Epoch 152/200] [D loss: 0.124511] [G loss:: 3.648633] [Elapsed Time: 3039.65s
    [Epoch 153/200] [D loss: 0.231062] [G loss:: 7.147243] [Elapsed Time: 3059.41s
    [Epoch 154/200] [D loss: 0.166175] [G loss:: 2.737486] [Elapsed Time: 3079.09s
    [Epoch 155/200] [D loss: 0.046858] [G loss:: 3.903417] [Elapsed Time: 3099.23s
    [Epoch 156/200] [D loss: 0.090757] [G loss:: 3.818349] [Elapsed Time: 3119.39s
    [Epoch 157/200] [D loss: 0.204279] [G loss:: 3.606827] [Elapsed Time: 3139.57s
    [Epoch 158/200] [D loss: 0.164988] [G loss:: 2.840378] [Elapsed Time: 3159.61s
    [Epoch 159/200] [D loss: 0.058896] [G loss:: 3.674007] [Elapsed Time: 3179.54s
    [Epoch 160/200] [D loss: 0.124426] [G loss:: 2.395360] [Elapsed Time: 3199.41s
    [Epoch 161/200] [D loss: 0.119471] [G loss:: 2.146797] [Elapsed Time: 3219.56s
    [Epoch 162/200] [D loss: 0.136473] [G loss:: 4.233829] [Elapsed Time: 3239.49s
    [Epoch 163/200] [D loss: 0.167176] [G loss:: 3.148930] [Elapsed Time: 3259.47s
    [Epoch 164/200] [D loss: 0.076814] [G loss:: 2.827775] [Elapsed Time: 3279.31s
    [Epoch 165/200] [D loss: 0.147114] [G loss:: 2.914271] [Elapsed Time: 3299.22s
    [Epoch 166/200] [D loss: 0.093404] [G loss:: 2.943054] [Elapsed Time: 3319.01s
    [Epoch 167/200] [D loss: 0.274170] [G loss:: 1.816038] [Elapsed Time: 3339.07s
    [Epoch 168/200] [D loss: 0.083843] [G loss:: 5.694256] [Elapsed Time: 3359.03s
    [Epoch 169/200] [D loss: 0.390972] [G loss:: 8.239223] [Elapsed Time: 3378.86s
    [Epoch 170/200] [D loss: 0.168692] [G loss:: 3.415112] [Elapsed Time: 3398.70s
    [Epoch 171/200] [D loss: 0.154357] [G loss:: 3.257585] [Elapsed Time: 3418.62s
    [Epoch 172/200] [D loss: 0.089064] [G loss:: 5.440307] [Elapsed Time: 3438.82s
    [Epoch 173/200] [D loss: 0.198712] [G loss:: 3.548514] [Elapsed Time: 3458.96s
    [Epoch 174/200] [D loss: 0.140017] [G loss:: 3.164138] [Elapsed Time: 3478.93s
    [Epoch 175/200] [D loss: 0.154613] [G loss:: 3.766619] [Elapsed Time: 3498.81s
    [Epoch 176/200] [D loss: 0.139869] [G loss:: 2.488168] [Elapsed Time: 3518.55s
    [Epoch 177/200] [D loss: 0.071725] [G loss:: 3.501382] [Elapsed Time: 3538.25s
    [Epoch 178/200] [D loss: 0.148876] [G loss:: 3.982516] [Elapsed Time: 3558.03s
    [Epoch 179/200] [D loss: 0.216380] [G loss:: 5.278347] [Elapsed Time: 3577.58s
    [Epoch 180/200] [D loss: 0.108764] [G loss:: 2.867591] [Elapsed Time: 3597.20s
    [Epoch 181/200] [D loss: 0.078762] [G loss:: 3.005085] [Elapsed Time: 3616.92s
    [Epoch 182/200] [D loss: 0.115718] [G loss:: 3.795631] [Elapsed Time: 3636.45s
    [Epoch 183/200] [D loss: 0.122913] [G loss:: 4.287529] [Elapsed Time: 3655.96s
    [Epoch 184/200] [D loss: 0.317448] [G loss:: 2.833395] [Elapsed Time: 3675.85s
    [Epoch 185/200] [D loss: 0.114451] [G loss:: 3.479819] [Elapsed Time: 3695.72s
    [Epoch 186/200] [D loss: 0.133126] [G loss:: 3.835314] [Elapsed Time: 3715.62s
    [Epoch 187/200] [D loss: 0.067699] [G loss:: 3.939270] [Elapsed Time: 3735.76s
    [Epoch 188/200] [D loss: 0.150934] [G loss:: 4.672352] [Elapsed Time: 3755.89s
    [Epoch 189/200] [D loss: 0.259734] [G loss:: 2.236391] [Elapsed Time: 3775.79s
    [Epoch 190/200] [D loss: 0.124371] [G loss:: 3.042009] [Elapsed Time: 3795.94s
    [Epoch 191/200] [D loss: 0.091835] [G loss:: 3.704115] [Elapsed Time: 3816.10s
    [Epoch 192/200] [D loss: 0.123711] [G loss:: 2.697129] [Elapsed Time: 3836.21s
    [Epoch 193/200] [D loss: 0.168072] [G loss:: 2.710114] [Elapsed Time: 3856.23s
    [Epoch 194/200] [D loss: 0.104286] [G loss:: 3.455408] [Elapsed Time: 3876.24s
    [Epoch 195/200] [D loss: 0.154701] [G loss:: 2.702538] [Elapsed Time: 3896.25s
    [Epoch 196/200] [D loss: 0.531704] [G loss:: 9.524725] [Elapsed Time: 3916.13s
    [Epoch 197/200] [D loss: 0.186313] [G loss:: 2.129881] [Elapsed Time: 3935.88s
    [Epoch 198/200] [D loss: 0.152626] [G loss:: 2.953154] [Elapsed Time: 3955.65s
    [Epoch 199/200] [D loss: 0.112756] [G loss:: 3.328232] [Elapsed Time: 3975.30s


## 5) Batch 200 학습할 때마다 sample한 data들 Visualization


```python
from IPython.display import Image

Image("92000.png")
```




    
![png](output_9_0.png)
    


