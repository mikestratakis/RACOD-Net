import torch
import torch.nn as nn
from src.Backbone.ResNet50 import ResNet50
from src.Backbone.ResNet50 import ResBlock50
from src.Backbone.SegFormerEncoder import MixVisionTransformer
from src.Decoder.PRDM import PRDM
from torchvision.models import resnet50, ResNet50_Weights
from typing import List

class RACOD(nn.Module):
    def __init__(self, img_size: int, embed_dims: List[int], num_heads: List[int], mlp_ratios: List[int], 
                 qkv_bias: bool, depths: List[int], sr_ratios: List[int], drop_rate: float, drop_path_rate: float,
                 decoder_channels: int, num_classes: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.MixVisionTransformerEncoder = MixVisionTransformer(img_size=img_size, embed_dims=embed_dims, 
                                                                num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, 
                                                                norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, 
                                                                drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.resnet = ResNet50(3, ResBlock50)
        self.PRDM = PRDM(embed_dims=embed_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        
        if self.training:
            print("*** Initializing Weights For ResNet50 Encoder ***")
            self.initialize_weights_resnet()      
            print("*** Initializing Weights For VisionTransformerEncoder ***")
            self.initialize_weights_vit()      
            
    def forward(self, x):
        x0 = self.resnet.layer0(x)              # (64, 102, 102)
        x1 = self.resnet.layer1(x0)             # (256, 102, 102)
        x2 = self.resnet.layer2(x1)             # (512, 51, 51)
        features = self.MixVisionTransformerEncoder(x)  # 4 feautures produced here from SegFormer
        segmentation = self.PRDM(features, x0, x1, x2)  # Returns 2 predictions both in size (1, 1, 408, 408)
        return segmentation
    
    def initialize_weights_resnet(self):
        weights = ResNet50_Weights.DEFAULT
        resnet50_pretrained = resnet50(weights=weights)
        pretrained_dict = resnet50_pretrained.state_dict()
        all_params = {}
        # we use this dictionary because of our ResNet50 implementation of layer0
        replace_dict = {
            'layer0.0.weight':'conv1.weight', 
            'layer0.2.weight': 'bn1.weight', 
            'layer0.2.bias' : 'bn1.bias', 
            'layer0.2.running_mean' : 'bn1.running_mean',
            'layer0.2.running_var' : 'bn1.running_var',
            'layer0.2.num_batches_tracked' : 'bn1.num_batches_tracked'
           }
        
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k and 'shortcut' not in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                if name in pretrained_dict:
                    v = pretrained_dict[name]
                    all_params[k] = v
                else:
                    all_params[k] = v
            elif '_2' in k and 'shortcut' not in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                if name in pretrained_dict:
                    v = pretrained_dict[name]
                    all_params[k] = v
                else:
                    all_params[k] = v
            # we use the below statement because of the word 'shortcut' instead of word 'downsample'
            # in our ResNet50 implementation
            elif 'shortcut' in k:
                name = k.split('shortcut')[0] + 'downsample' + k.split('shortcut')[1]
                if name in pretrained_dict:
                    v = pretrained_dict[name]
                    all_params[k] = v
                else:
                    if '_1' in name:
                        new_name = name.replace('_1', '')
                        v = pretrained_dict[new_name]
                        all_params[k] = v
                    elif '_2' in name:
                        new_name = name.replace('_2', '')
                        v = pretrained_dict[new_name]
                        all_params[k] = v
            elif k in replace_dict:
                name = replace_dict[k]
                v = pretrained_dict[name]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        print('[INFO] Weights Initialized For Resnet50 From torchvision.models Trained In IMAGENET1K_V2')
    
    def initialize_weights_vit(self):
        PATH_TO_PRETRAINED_WEIGHTS = './Pretrained_Weights_SegFormer/mit_b4.pth'
        CheckPoint = torch.load(PATH_TO_PRETRAINED_WEIGHTS)
        
        parameters_altered = 0
        all_params = {}
        for k_model, v_model in self.MixVisionTransformerEncoder.state_dict().items():
            if k_model in CheckPoint.keys():
                all_params[k_model] = CheckPoint[k_model]
                parameters_altered += 1
            else:
                all_params[k_model] = v_model
        
        assert len(all_params.keys()) == len(self.MixVisionTransformerEncoder.state_dict().keys())
        self.MixVisionTransformerEncoder.load_state_dict(all_params)
        print('[INFO] Weights Initialized For MixVisionTransformerEncoder From mit_b4 Weights Trained In ImageNet-1K')