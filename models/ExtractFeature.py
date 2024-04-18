import timm
from mmpretrain import get_model

import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractionSwin(nn.Module):
    
    def __init__(self, weights=None, pre=True, device='cpu'):
        super(FeatureExtractionSwin, self).__init__()
        '''
            swin-base_in21k-pre-3rdparty_in1k
            swin-large_in21k-pre-3rdparty_in1k
        '''
        self.model = get_model(weights, 
                               pretrained=pre, 
                               device=device, 
                               backbone=dict(out_indices=(0, 1, 2, 3)))
        
        for param in self.model.parameters():
            param.requires_grad = False

        # self.pad = nn.ReflectionPad2d(padding=1)
        
    def forward(self, x):
        features = self.model.extract_feat(x, stage='backbone')
        # features = [self.pad(f) for f in features]
        return list(features)

class Feature_Extractor_Resnet50(nn.Module):
    
    def __init__(self, weights=None, gradient=False):
        super(Feature_Extractor_Resnet50, self).__init__()
        self.model = models.resnet18(weights=weights)
        # self.model = models.wide_resnet50_2(weights=weights)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        if not gradient:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x0 = self.layer0(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4] # 元组输出
    
class FeatureAggregation(nn.Module):
    '''
        resize(256) + concat + averagepool(N)
    '''
    def __init__(self, feature_size=64, scale_size=64, pool=False):
        super(FeatureAggregation, self).__init__()
        
        if pool:
            self.pool = nn.AdaptiveAvgPool2d(output_size=(scale_size, scale_size))
        else:
            self.pool = False

        self.feature_size = feature_size
    
    def forward(self, inputs):
        resized_features = [nn.functional.interpolate(feature, 
                                                      size=self.feature_size, 
                                                      mode='bilinear', 
                                                      align_corners=True) for feature in inputs]
        features = torch.cat(resized_features, dim=1)
        if self.pool:
            features = self.pool(features)
        return features
    
def Add_Noise(inputs):
    '''
        add gaussian noise [B,C,H,W]
    '''
    noise = torch.randn_like(inputs).to(inputs.device)
    inputs_noise = inputs + 0.1*noise
    return inputs_noise

def l2_normalize(input, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)

if __name__ == '__main__':

    import timm
    timm_models = timm.list_models(pretrained=True)
    for model in timm_models:
        if 'pvt' in model:
            print(model)

    inputs = torch.randn(4, 3, 256, 256)
    FESwin = FeatureExtractionSwin(weights='swin-large_in21k-pre-3rdparty_in1k')
    outputs = [inputs] + FESwin(inputs)
    for idx, i in enumerate(outputs):
        print(idx, i.shape, i.requires_grad)

    """
    0 torch.Size([4, 192, 64, 64]) False
    1 torch.Size([4, 384, 32, 32]) False
    2 torch.Size([4, 768, 16, 16]) False
    3 torch.Size([4, 1536, 8, 8]) False
    """

    FEResnet = Feature_Extractor_Resnet50(weights='IMAGENET1K_V1')
    outputs = [inputs] + FEResnet(inputs)
    for idx,i in enumerate(outputs):
        print(idx, i.shape, i.requires_grad)

    """
    0 torch.Size([4, 64, 64, 64]) False
    1 torch.Size([4, 128, 32, 32]) False
    2 torch.Size([4, 256, 16, 16]) False
    3 torch.Size([4, 512, 8, 8]) False
    """