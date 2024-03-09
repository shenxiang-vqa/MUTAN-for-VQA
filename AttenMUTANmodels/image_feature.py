import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet152_Weights
from torchsummary import summary
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder,self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        for param in resnet.parameters():
            param.requires_grad = False
        self.cnn = resnet
    def forward(self,image):
        features = self.cnn(image) #[batch,2048,14,14]
        return features
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ImageEncoder().to(device)
#     image = torch.randn(1,3,448,448).to(device)
#     out = model(image)
#     print(out)
#     print(out.size())
