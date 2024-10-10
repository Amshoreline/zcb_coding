# Pytorch常用代码

对应Pytorch 1.x版本，相关包

``` python
import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image as Image
import torch
import torchvision
```

## 1.基础配置

检查Pytorch版本

```python
torch.__version__               # Pytorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
```

CUDA是NVIDIA提出的，在GPU上的编程语言。cuDNN是使用CUDA语言编写的深度神经网络库，它为深度神经网络的常见操作提供GPU加速功能。

固定随机种子

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

指定程序在特定GPU上跑的两种方式：

1. 在命令行指定环境变量

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python main.py
    ```

2. 在代码中指定

    ```python
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    ```

判断是否有CUDA支持

```python
torch.cuda.is_available()
```

cuDNN benchmark模式，打开Benchmark模式会提升计算速度，但是由于计算的随机性，每次网络前向传播结果略有差异

```python
torch.backends.cudnn.benchmark = True
```

如果想要避免这种结果波动，再设置一下：

```python
torch.backends.cudnn.deterministic = True
```

清除GPU显存

```python
torch.cuda.empty_cache()
```

## 2.torch tensor

基本信息

```python
tensor.type()   # Data type(str)
tensor.dtype    # Data type
tensor.size()   # Shape of the tensor
tensor.dim()    # Number of dimensions
```

数据类型转换

```python
torch.set_default_tensor_type(torch.FloatTensor)
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.double()
tensor = tensor.int()
tensor = tensor.long()
```

改变数组形状

```python
shape = (2, 2)
tensor.view(*shape)     # share the same memory
tensor.reshape(*shape)  # return a copy
```

与np.ndarray转换

```python
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray)
tensor = torch.from_numpy(ndarray.copy()).float()   # if ndarray has negative step
tensor = torch.Tensor(ndarray)
```

与PIL.Image转换

```python
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255
    ).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way
image = torchvision.transforms.ToPILImage()(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))
    ).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))  # Equivalently way
tensor = torchvision.transforms.ToTensor()(PIL.Image.open(path))  # Equivalently way
```

np.ndarray 与 PIL.Image 转换

```python
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astypde(np.uint8))
# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
```

从只包含一个元素的张量中提取值

```python
value = tensor.item()
```

打乱顺序

```python
tensor = tensor[torch.randperm(tensor.size(0))]
```

水平翻转

```python
# Assume tensor has shape N * D * H * W.
tensor = tensor[:, :, :, torch.arange(tensor.size(3) - 1, -1, -1).long()]
```

复制tensor

```python
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach().clone()   # |        New         |          No                |
```

拼接tensor：stack会无中生维

```python
tensor = torch.cat(tensor_list, dim=0)
tensor = torch.stack(tensor_list, dim=0)
```

转换成one-hot编码

```python
N = tensor.size(0)
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
```

得到非零/零元素

```python
torch.nonzero(tensor)               # index of non-zero elements
torch.nonzero(tensor == 0)          # index of zero elements
torch.nonzero(tensor).size(0)       # number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # number of zero elements
```

tensor扩展

```python
tensor.reshape(*tensor.size(), 1, 1).expand(*tensor.size(), 7, 7)
```

矩阵乘法

```python
# Matrix multiplication: (m * n) * (n * p) -> (m * p).
result = torch.mm(tensor1, tensor2)
# Batch matrix multiplication: (b * m * n) * (b * n * p) -> (b * m * p).
result = torch.bmm(tensor1, tensor2)
# Element-wise multiplication.
result = tensor1 * tensor2
```

两组数据之间两两欧氏距离

```python
# x1.size() = (m, d)
x1 = x1.unsqueeze(x1, dim=1).expand(x1.size(0), x2.size(0), x1.size(1))
# x2.size() = (n, d)
x2 = x2.unsqueeze(x2, dim=0).expand(x1.size(0), x2.size(0), x2.size(1))
dist = torch.sqrt(torch.sum((x1 - x2) ** 2), dim=2)
```

padding

```python
tensor = torch.ones(2, 2, 2)
torch.nn.functional.pad(tensor, (0, 1, 0, 1), value=-1) # size = (2, 3, 3)
```