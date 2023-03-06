import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out) + self.sigmoid(max_out)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# New try 
class PRDM(nn.Module):
    def __init__(self, embed_dims, decoder_channels, num_classes):
        super(PRDM, self).__init__()
        assert min(embed_dims) == embed_dims[0]
        self.feature_strides = embed_dims
        self.num_classes = num_classes
        embedding_dim = decoder_channels
        self.conv_128 = BasicConv2d(128, embedding_dim, 1)
        self.conv_512 = BasicConv2d(512, embedding_dim, 1)
        self.conv_convert_64_384 = BasicConv2d(64, embedding_dim // 2, kernel_size=3, stride=1, padding=1,dilation=1)
        self.conv_convert_320_384 = BasicConv2d(320, embedding_dim // 2, kernel_size=3, stride=1, padding=1,dilation=1)
        self.conv_convert_320_512 = BasicConv2d(320, 512, kernel_size=3, stride=1, padding=1,dilation=1)
        self.conv_convert_1024_768 = BasicConv2d(1024, embedding_dim, kernel_size=3, stride=1, padding=1,dilation=1)
        
        self.conv_same = BasicConv2d(in_channels=embedding_dim, out_channels=embedding_dim, 
                                     kernel_size=3, stride=1, padding=1,dilation=1)
        
        self.conv_restore_double = BasicConv2d(in_channels=2*embedding_dim, out_channels=embedding_dim, 
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.conv_double_same = BasicConv2d(in_channels=2*embedding_dim, out_channels=2*embedding_dim, 
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.channelattn = ChannelAttention(embedding_dim)
        #self.relu = nn.ReLU(inplace=True)
        self.linear_pred_top = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.linear_pred_final = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, features, x0, x1, x2):
        c1, c2, c3, c4 = features[0], features[1], features[2], features[3]
        
        ############## Fusing C3 With X2 #############
        conv_c3 = self.conv_convert_320_512(F.interpolate(c3, size=x2.size()[2:], mode='bilinear', align_corners=False))        
        _c3_x2 = conv_c3 * x2 # (BS, 512, 57, 57)
        mixed_features = self.conv_convert_1024_768(torch.cat((_c3_x2, conv_c3), dim=1)) # (BS, 768, 57, 57)
        
        ############## Decoder on C2-C4 ##############
        _c2 = self.conv_128(c2) # from (BS, 128, 57, 57) => (BS, 768, 57, 57)
        _c3 = self.conv_same(F.interpolate(mixed_features, size=c3.size()[2:], mode='bilinear', align_corners=False)) # (BS, 768, 29, 29)
        _c4 = self.conv_512(c4) # from (BS, 512, 15, 15) => (BS, 768, 15, 15)
        _c4_upsample2 = self.conv_same(F.interpolate(_c4, size=c3.size()[2:], mode='bilinear', align_corners=False)) # (BS, 768, 29, 29)
        
        _c4_c3 = self.conv_double_same(torch.cat((_c4_upsample2 * _c3, _c4_upsample2), dim=1)) # (BS, 1536, 29, 29)
        _c4_c3 = self.conv_restore_double(F.interpolate(_c4_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)) # (768, 57, 57)
        _c4_upsample4 = self.conv_same(F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)) # (768, 57, 57)
        _c4_c3_c2 = _c4_upsample4 * _c2 * _c4_c3 # (768, 57, 57)
        
        top_features = self.conv_restore_double((torch.cat((_c4_c3_c2,  mixed_features), dim=1))) # (768, 57, 57)
        final_output_top = F.interpolate(self.linear_pred_top(top_features), scale_factor=8, mode='bilinear', align_corners=False) # (1, 1, 456, 456)

        ############## Decoder on C1-X0-X1 ##############
        x01 = self.conv_convert_320_384(torch.cat((x0, x1), dim=1))       # (BS, 384, 114, 114)
        #_c1_x0 = self.conv_convert_64_384(c1 * x0)                        # (BS, 384, 114, 114)
        final_bottom = self.conv_same(torch.cat((self.conv_convert_64_384(c1 * x0) * x01 , x01), dim=1)) # (BS, 768, 114, 114)
        final_bottom = self.conv_same(F.interpolate(final_bottom, size=c2.size()[2:], mode='bilinear', align_corners=False))# (BS, 768, 57, 57)
        final_bottom = self.conv_restore_double(torch.cat((mixed_features * final_bottom, mixed_features), dim=1)) # (BS, 768, 57, 57)

        ############## Channel Attention Between Top and Bottom  Features ##############
        add_features = top_features + final_bottom # (BS, 768, 57, 57)
        final_output = self.channelattn(add_features) * add_features # (BS, 768, 57, 57)
        mixed_features = self.conv_restore_double(torch.cat((_c4_c3, final_bottom), dim=1))  # (BS, 768, 57, 57)
        final_output = self.channelattn(mixed_features) * final_output # (BS, 768, 57, 57)
        final_output = F.interpolate(self.linear_pred_final(final_output), scale_factor=8, mode='bilinear', align_corners=False) # (1, 1, 456, 456)
        
        return final_output_top, final_output