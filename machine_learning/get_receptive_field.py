import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt


class NaiveModel(torch.nn.Module):
    
    # A naive model with 4 conv2ds and 4 pool2ds.
    # Its receptive field is ((((((1 x 2) + 2) x 2 + 2) x 2) + 2) x 2 + 2) = 46
    def __init__(self, ):
        super().__init__()
        for i in range(4):
            conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
            self.__setattr__(f'conv{i}', conv)
            bn = torch.nn.BatchNorm2d(3)
            self.__setattr__(f'bn{i}', bn)
        self.pool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        for i in range(4):
            x = self.__getattr__(f'conv{i}')(x)
            x = self.__getattr__(f'bn{i}')(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
        return x


class VisionModel(torch.nn.Module):
    
    def __init__(self, model_name):
        super().__init__()
        model = models.__dict__[model_name](pretrained=False)
        # Remove AdaptiveAvgPool2d and Linear
        layers = [child for child in model.children()][: -2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)


def convert_model(module):
    mod = module
    if isinstance(module, torch.nn.MaxPool2d):
        assert module.dilation == 1 and module.ceil_mode == False
        mod = torch.nn.AvgPool2d(module.kernel_size, module.stride, module.padding)
    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
    return mod


def init_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = 'replicate'
            size = module.weight.size()
            torch.nn.init.constant_(module.weight, 1. / size[-3] / size[-2] / size[-1])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.running_var)
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.running_mean)
            torch.nn.init.zeros_(module.bias)


def get_receptive_field(model, input_size=512):
    model = convert_model(model)
    print(model)
    init_model(model)
    model.eval()
    x = torch.ones(1, 3, input_size, input_size, requires_grad=True)
    output = model(x)
    output = torch.mean(torch.mean(output, dim=1), dim=0)
    print(output[0, 0])
    output_size = output.size(0)
    output[output_size // 2, output_size // 2].backward()
    grad = x.grad.detach().numpy()[0, 0]
    size_row = np.sum(np.max(grad, axis=1) > 0)
    size_col = np.sum(np.max(grad, axis=0) > 0)
    print(f'Receptive field is ({size_col}, {size_row})')
    grad /= np.max(grad) + 1e-12
    plt.imshow(grad, cmap='gray')
    plt.show()


if __name__ == '__main__':
    get_receptive_field(NaiveModel())
    get_receptive_field(VisionModel('vgg16'))
    get_receptive_field(VisionModel('resnet18'))
    get_receptive_field(VisionModel('resnext50_32x4d'))
