import torch.nn as nn

class ResBlock50(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        
        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels//4 if downsample else in_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(out_channels//4 if downsample else in_channels, out_channels//4 if downsample else in_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(out_channels//4 if downsample else in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels//4 if downsample else in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels//4 if downsample else in_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels//4)
            self.bn2 = nn.BatchNorm2d(out_channels//4)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
            
        if downsample or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        #input = nn.ReLU()(self.bn2(self.conv2(input)))
        # An operation here refers to a convolution a batch normalization and a ReLU activation to an input, 
        # except the last operation of a block, that does not have the ReLU.
        # https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = self.bn3(self.conv3(input))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet50(nn.Module):
    def __init__(self, in_channels, resblock50):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock50(64, 256, downsample=False),
            resblock50(256, 256, downsample=False),
            resblock50(256, 256, downsample=False),
        )

        self.layer2 = nn.Sequential(
            resblock50(256, 512, downsample=True),
            resblock50(512, 512, downsample=False),
            resblock50(512, 512, downsample=False),
            resblock50(512, 512, downsample=False),
        )

        self.layer3_1 = nn.Sequential(
            resblock50(512, 1024, downsample=True),
            resblock50(1024, 1024, downsample=False),
            resblock50(1024, 1024, downsample=False),
            resblock50(1024, 1024, downsample=False),
            resblock50(1024, 1024, downsample=False),
            resblock50(1024, 1024, downsample=False),
        )
        
        self.layer4_1 = nn.Sequential(
            resblock50(1024, 2048, downsample=True),
            resblock50(2048, 2048, downsample=False),
            resblock50(2048, 2048, downsample=False),
        )
        
    def forward(self, input):
        input = self.layer0(input)
        x_0 = input
        input = self.layer1(input)
        x_1 = input
        input = self.layer2(input)
        x_2 = input
        input = self.layer3_1(input)
        x_3 = input
        input = self.layer4_1(input)
        x_4 = input
        
        return x_0, x_1, x_2, x_3, x_4