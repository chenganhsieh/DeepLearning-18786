import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        My custom ResidualBlock

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        
        identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class MyResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        """
        My custom ResNet.

        [input]
        * in_channels  : input channel number
        * num_classes  : number of classes

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.resblock1 = ResidualBlock(64, 128, kernel_size=3, stride=2)
        self.resblock2 = ResidualBlock(128, 256, kernel_size=3, stride=2)
        self.resblock3 = ResidualBlock(256, 512, kernel_size=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x, return_embed=False):
        """
        Forward path.

        [input]
        * x             : input data
        * return_embed  : whether return the feature map of the last conv layer or not

        [output]
        * output        : output data
        * embedding     : the feature map after the last conv layer (optional)
        
        [hint]
        * See the instruction PDF for network details
        * You want to set return_embed to True if you are dealing with CAM
        """

        # ----- TODO -----
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        
        # Optional embedding output
        embedding = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if return_embed:
            return x, embedding
        else:
            return x


def init_weights_kaiming(m):

    """
    Kaming initialization.

    [input]
    * m : torch.nn.Module

    [hint]
    * Refer to the course slides/recitations for more details
    * Initialize the bias term in linear layer by a small constant, e.g., 0.01
    """

    if isinstance(m, nn.Conv2d):
        # ----- TODO -----
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # ----- TODO -----
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.01)


if __name__ == "__main__":

    # set model
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)
    
    # sanity check
    input = torch.randn((64, 3, 32, 32), requires_grad=True)
    output = net(input)
    print(output.shape)