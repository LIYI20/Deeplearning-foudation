## Pytorch学习

####  1.dir()和help()函数

> dir()可以看到package中的不同子包或方法
>
> help()可以找到官方文档的说明

#### 2.加载数据

##### Dataset

- 提供方法获取数据和label

1.获得每个数据和label

2.获得数据数量

![image-20240330154112451](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330154112451.png)

![image-20240330154506736](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330154506736.png)

- os.path.join是 将两个路径连接起来得到一个路径
- os.listdir是将path中的所有文件转换为一个list列表
- init函数类似构造函数

##### Dataloader

- 为网络提供不同的数据形式

#### 2.梯度下降法

- Gradient Descent
- x'=x-梯度*lr

![image-20240330161232971](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330161232971.png)

- 让loss更小，更好的拟合，从而预测更准，lr是learningRate

![image-20240330162152242](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330162152242.png)

- 用梯度求最小值

![image-20240330162410431](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330162410431.png)

- 循环，返回最后一组

![image-20240330162704488](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330162704488.png)

#### 3.数据类型

- tensor是张量，是矩阵的扩展和延伸

![image-20240330170557144](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330170557144.png)

- 用torch.IntTensor,torch.LongTensor...

- GPU tensor用的是torch.cuda...

![image-20240330170428694](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330170428694.png)

```
torch.tensor(1.0)//标量
torch.tensor([1,2])//一维向量
torch.FloatTensor(1)//size为1的向量。随机赋值
```

![image-20240330171353324](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330171353324.png)

![image-20240330171747881](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330171747881.png)

#### 4.transform

- ToTensor（），将图片，PILimg或nparray类型转为tensor
- Normalize（）归一化，输入均值和标准差
- ![image-20240330205620284](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330205620284.png)
- Resize（）将图片的大小进行改变，只输入一个参数的话，进行等比缩放

![image-20240330205340252](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330205340252.png)

- Compose（）将多个transform对象进行组合，按顺序
- ![image-20240330210141877](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330210141877.png)

![image-20240330210430779](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240330210430779.png)

- RandomCrop随机裁剪

#### 5.torchvision中的数据集使用

```python
from sympy import true
import torchvision
dataset_tranform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
from torch.utils.tensorboard import SummaryWriter
train_set=torchvision.datasets.CIFAR10(root="./data_set",train=True,transform=dataset_tranform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./test_set",train=False,transform=dataset_tranform,download=True)


# print(test_set[0])
# image,target=test_set[0]
# print(image)
# print(target)
# print(test_set.classes)
# print(test_set.classes[target])
# image.show()
writer=SummaryWriter("pytorch_study")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)
writer.close()


```

#### 6.DataLoader

```python
from sympy import false
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
test_data=torchvision.datasets.CIFAR10("./data_set",train=False,transform=torchvision.transforms.ToTensor())
#batch_size是每次取的数量，shuffle是随机取，drop_last是最后不满足batch_size是否舍去，
#data_loader是每次取的list的集合，list组成可以看后面
test_loader=DataLoader( dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
print(test_data[0])
img,target=test_data[0]
#img在tensor中的存储是（C，W，H），即通道数，宽，高，在确定的通道，宽和高的位置上存放的是像素值，
#不过transform的时候将像素除以255，转化到了[0-1]之间
print(img.shape)
#target就是label的索引
writer=SummaryWriter("pytorch_study")
step=0
for data in test_loader:
    imgs,targets=data
    #imgs是4张组合成的tensor，会多一个维度用来储存img数目
    #targets是放的4个target组成的数组
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_images",imgs,step)
    step=step+1
writer.close()
    
```

#### 7.nn.Moudle

- neural network

![image-20240331105717478](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331105717478.png)

- forward函数，conv卷积，relu非线性

![image-20240331105617453](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331105617453.png)

- forward函数会在__ call__()函数中调用，所以在传参的时候就会执行forward

```python
from turtle import forward
from torch import nn, tensor
import torch

class Modle(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,input):
        output=input+1
        return output
    
modle=Modle()
input=tensor(1)
output=modle(input)
print(output)

```

#### 8. 卷积

- conv1d是一维的，conv2d是二维的，以此类推
- stride是每次卷积核移动的步长，默认横竖都是1，只传一个数的话横竖都是这个数，也可以传一个tuple（sH，sW）分别设置
- padding，填充tensor，如果padding为1，则四周都填一格，也可以传一个tuple（pH，pW）分别设置

![image-20240331111149357](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331111149357.png)

- padding

![image-20240331113441143](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331113441143.png)

```python
import torch
import torch.nn.functional as F


input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

input =torch.reshape(input,(1,1,5,5))
#进行卷积的时候，要对input进行处理
#分别是batch_size,通道,W,H
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3,))

print(input.shape)
print(kernel.shape)

output0=F.conv2d(input,kernel,stride=1)
print("output0")
print(output0)

output1=F.conv2d(input,kernel,stride=2)
print("output1")
print(output1)

output2=F.conv2d(input,kernel,stride=1,padding=1)
print("output2")
print(output2)


```

- 结果

![image-20240331113322971](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331113322971.png)

#### 9.卷积层

![image-20240331133445307](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331133445307.png)

- in_channel 输入的channel数
- out_channel输出的channel数

- reshape函数

  ![image-20240331142059292](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331142059292.png)

![image-20240331142120267](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331142120267.png)

```python
from turtle import forward
import torch
import torchvision
from torch import  nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set=torchvision.datasets.CIFAR100("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader=DataLoader(data_set,batch_size=64)
class Module(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=Conv2d(3,6,3,1,0)


    def forward(self,x):
        x=self.conv1(x)
        return x
    
module=Module()
print(module)

writer=SummaryWriter("./pytorch_study")
step=0
for data in data_loader:
    imgs,targets=data
    output=module(imgs)
    print(imgs.shape)
    print(output.shape)
    output=torch.reshape(output,(-1,3,30,30))
    print(output.shape)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step=step+1

writer.close()
```

![image-20240331143535072](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331143535072.png)

![image-20240331143600751](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240331143600751.png)

#### 10.pooling layer（池化层）

- 和卷积层类似，池化层也有池化核
- ceil model为true会保留不满尺寸的元素，为false不会保留
- pooling layer中的strid默认是kernel_size
- apooling layer 可以减少data量，取一部分中的最大值

```python
import torch
from torch import nn
from torch.nn import MaxPool2d
input =torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
input =torch.reshape(input,(-1,1,5,5))
print(input.shape)

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output
net1=net()
output=net1(input)
print(output)    

```

![image-20240402144224533](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402144224533.png)

#### 11.非线性激活

##### Relu

- Relu(input,inplace)
- inplace=true时，input是-1，input会替换为0，而false不会替换，默认为false

#### 12.线性层

- 先把tensor转为一维，也可以用flatten来实现
- Linear(input_future,output_future)把input_future 个转为output_future个

![image-20240402150824907](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402150824907.png)

![全连接层： 权值初始化_怎样初始化全连接层权重-CSDN博客](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9nYXJkZW4tbHUtb3NzLm9zcy1jbi1iZWlqaW5nLmFsaXl1bmNzLmNvbS9pbWFnZXMvMjAyMDA1MTkxMTIwMjYucG5n?x-oss-process=image/format,png)

#### 13.CIFAR10测试

![image-20240402151818867](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402151818867.png)

```python
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Sequential

class Net(nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    #     self.conv1=Conv2d(3,32,5,padding=2)
    #     self.max_pool1=MaxPool2d(2)
    #     self.conv2=Conv2d(32,32,5,padding=2)
    #     self.max_pool2=MaxPool2d(2)
    #     self.conv3=Conv2d(32,64,5,padding=2)
    #     self.max_pool3=MaxPool2d(2)
    #     self.flatten=Flatten()
    #     self.linear1=Linear(1024,64)
    #     self.linear2=Linear(64,10)

    # def forward(self,x):
    #     x=self.conv1(x)
    #     x=self.max_pool1(x)
    #     x=self.conv2(x)
    #     x=self.max_pool2(x)
    #     x=self.conv3(x)
    #     x=self.max_pool3(x)
    #     x=self.flatten(x)
    #     x=self.linear1(x)
    #     x=self.linear2(x)
    #     return x
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode1=Sequential(
        Conv2d(3,32,5,padding=2),
        MaxPool2d(2),
        Conv2d(32,32,5,padding=2),
        MaxPool2d(2),
        Conv2d(32,64,5,padding=2),
        MaxPool2d(2),
        Flatten(),
        Linear(1024,64),
        Linear(64,10)
        )

    def forward(self,x):
        x=self.mode1(x)
        return x


net=Net()
print(net)

#检查
input=torch.ones((64,3,32,32))
output=net(input)
print(output.shape)

```

#### 13.损失函数和反向传播

- loss指的是实际与预期的差距值，loss越小越好

- L1Loss(input,target)计算input和target的差值的平均值，可以设定reduction=‘sum’变成计算差值的和，input要有batch_size这个维度

- MSELoss（input，target）计算差值平方和再平均
- CrossEntropyLoss（input，target）交叉熵损失input（N，C）output（N）
- 利用loss来进行反向传播，得到梯度

![image-20240402194001549](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402194001549.png)

![image-20240402194840726](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402194840726.png)

#### 14.优化器（optim）

- lr learning rate

- 优化器会根据梯度进行参数的调整，让loss更小

  ![image-20240402200232853](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402200232853.png)

#### 15.量化

- 非对称量化相较于对称量化有一个zero_point
- 对称量化可以看作是选取作为zero_point

![image-20240508134634207](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240508134634207.png)

- 对称量化

![image-20240508134812892](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240508134812892.png)

- 量化流程

![image-20240508134850184](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240508134850184.png)

- 总流程

![image-20240508135759448](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240508135759448.png)

![image-20240508142144693](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240508142144693.png)
