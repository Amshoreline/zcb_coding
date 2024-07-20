# Python Hook学习笔记

> <https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904>

Hook定义：在一个特定事件后执行的方法

Pytorch hooks注册于某个Tensor或nn.Module，触发于forward或backward时:

```python
from torch import nn. Tensor

def module_hook(module: nn.Module, data: Tensor, output: Tensor):
    # For nn.Module objects only

def tensor_hook(grad: Tensor):
    # For Tensor objects only
    # Only executed during the *backward* pass!
```

## Verbose Model Execution

```python
import torch
from torch import nn. Tensor
from torchvision.models import resnet50


class VerboseExecution(nn.Module):
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(
                    f'{layer.__name__}: {output.shape}'
                )
            )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def test():
    verbose_resnet = VerboseExecution(resnet50())
    dummy_input = torch.ones(10, 3, 224, 224)
    _ = verbose_resnet(dummy_input)
    # conv1: torch.Size([10, 64, 112, 112])
    # bn1: torch.Size([10, 64, 112, 112])
    # relu: torch.Size([10, 64, 112, 112])
    # maxpool: torch.Size([10, 64, 56, 56])
    # layer1: torch.Size([10, 256, 56, 56])
    # layer2: torch.Size([10, 512, 28, 28])
    # layer3: torch.Size([10, 1024, 14, 14])
    # layer4: torch.Size([10, 2048, 7, 7])
    # avgpool: torch.Size([10, 2048, 1, 1])
    # fc: torch.Size([10, 1000])

```

## Feature Extraction

```python
import torch
from torch import nn, Tensor
from torchvision.models import resnet50
from typing import Dict, Iterable, Callable


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_names: Iterable[str]):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features = {
            layer_name : torch.empty(0) for layer_name in layer_names
        }
        for layer_name in layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.save_outputs_hook(layer_name))

    def save_outputs_hook(self, layer_name: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_name] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def test():
    dummy_input = torch.ones(10, 3, 224, 224)
    resnet_features = FeatureExtractor(resnet50(), ["layer4", "avgpool"])
    features = resnet_features(dummy_input)
    # {
    #     'layer4': torch.Size([10, 2048, 7, 7]),
    #      'avgpool': torch.Size([10, 2048, 1, 1])
    # }
    print({name: output.shape for name, output in features.items()})
```

## Gradient Clipping

```python
import torch
from torch import nn, Tensor
from torchvision.models import resnet50


def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))
    return model

def test():
    clipped_resnet = gradient_clipper(resnet50(), 0.001)
    pred = clipped_resnet(dummy_input)
    loss = pred.log().mean()
    loss.backward()
    print(clipped_resnet.fc.bias.grad[: 25])
```