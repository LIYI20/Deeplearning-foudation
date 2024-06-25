#### torch.mean()

- 求平均数
- mean()函数的参数：dim=0,按列求平均值，返回的形状是（1，列数）；dim=1,按行求平均值，返回的形状是（行数，1）,默认不设置dim的时候，返回的是所有元素的平均值

![image-20240402101807081](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240402101807081.png)

```python
import torch

x1 = torch.Tensor([1, 2, 3, 4])
x2 = torch.Tensor([[1],
                   [2],
                   [3],
                   [4]])
x3 = torch.Tensor([[1, 2],
                   [3, 4]])
y1 = torch.mean(x1)
y2 = torch.mean(x2,0)#按列
y3 = torch.mean(x3,1)#按行
print(y1)
print(y2)
print(y3)

```

#### torch.argmax

- dim的取值为[-2, 1]之间，只能取整，有四个数，0和-2对应，得到的是每一列的最大值，1和-1对应，得到的是每一行的最大值。如果参数中不写dim，则得到的是张量中最大的值对应的索引（从0开始）

```python
x = torch.randn(3, 5)
print(x)
print(torch.argmax(x))
print(torch.argmax(x, dim=0))
print(torch.argmax(x, dim=-2))
print(torch.argmax(x, dim=1))
print(torch.argmax(x, dim=-1))



output:
tensor([[-1.0214,  0.7577, -0.0481, -1.0252,  0.9443],
        [ 0.5071, -1.6073, -0.6960, -0.6066,  1.6297],
        [-0.2776, -1.3551,  0.0036, -0.9210, -0.6517]])
tensor(9)
tensor([1, 0, 2, 1, 1])
tensor([1, 0, 2, 1, 1])
tensor([4, 4, 2])
tensor([4, 4, 2])
```

#### criterion

![image-20240403165109904](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240403165109904.png)

### 2.Quantization

#### 2.1 tensor量化

quantize_per_tensor函数就是使用给定的scale和zp来把一个float tensor转化为quantized tensor，后文你还会遇到这个函数。通过上面这几个数的变化，你可以感受到，量化tensor，也就是xq，和fp32 tensor的关系大概就是:

```text
xq = round(x / scale + zero_point)
```

```python
>>> x = torch.rand(2,3, dtype=torch.float32) 
>>> x
tensor([[0.6839, 0.4741, 0.7451],
        [0.9301, 0.1742, 0.6835]])

>>> xq = torch.quantize_per_tensor(x, scale = 0.5, zero_point = 8, dtype=torch.quint8)
tensor([[0.5000, 0.5000, 0.5000],
        [1.0000, 0.0000, 0.5000]], size=(2, 3), dtype=torch.quint8,
       quantization_scheme=torch.per_tensor_affine, scale=0.5, zero_point=8)

>>> xq.int_repr()
tensor([[ 9,  9,  9],
        [10,  8,  9]], dtype=torch.uint8)
```

- 反量化，会有精度损失

```python
# xq is a quantized tensor with data represented as quint8
>>> xdq = xq.dequantize()
>>> xdq
tensor([[0.5000, 0.5000, 0.5000],
        [1.0000, 0.0000, 0.5000]])
```

#### 2.2 **Post Training Dynamic Quantization**

- Post：也就是训练完成后再量化模型的权重参数；
- Dynamic：也就是网络在前向推理的时候动态的量化float32类型的输入。

Dynamic Quantization使用下面的API来完成模型的量化：

```python
torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None,
```

#### 2.3**Post Training Static Quantization**

##### **2.3.1 fuse_model**

![image-20240507171709327](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507171709327.png)

```python
DEFAULT_OP_LIST_TO_FUSER_METHOD : Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}
```

##### 2.3.2 **设置qconfig**

![image-20240507171937697](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507171937697.png)

##### 2.3.3 prepare

![image-20240507172139467](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507172139467.png)

#### 2.4 **QAT（Quantization Aware Training）**

##### 2.4.1 qconfig and fuse

![image-20240507172551425](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507172551425.png)

![image-20240507172710995](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507172710995.png)

##### 2.4.2prepare_qat

![image-20240507172834445](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507172834445.png)

![image-20240507172859874](C:\Users\w\AppData\Roaming\Typora\typora-user-images\image-20240507172859874.png)