import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def MyFConv2D(input, weight, bias=None, stride=1, padding=0):
    
    """
    My custom Convolution 2D calculation.

    [input]
    * input    : (batch_size, in_channels, input_height, input_width)
    * weight   : (you have to derive the shape :-)
    * bias     : bias term
    * stride   : stride size
    * padding  : padding size

    [output]
    * output   : (batch_size, out_channels, output_height, output_width)
    """

    assert len(input.shape) == len(weight.shape)
    assert len(input.shape) == 4
    
    ## padding x with padding parameter 
    ## HINT: use torch.nn.functional.pad()
    # ----- TODO -----
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    # x_pad = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)

    ## Derive the output size
    ## Create the output tensor and initialize it with 0
    # ----- TODO -----
    output_height = ((input_height + 2 * padding - kernel_height) // stride) + 1
    output_width  = ((input_width + 2 * padding - kernel_width) // stride) + 1
    
    # ref: https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
    # x_unfolded = F.unfold(input, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)
    # weight_reshaped = weight.view(out_channels, -1)
    # output_2d = weight_reshaped @ x_unfolded
    # if bias is not None:
    #     output_2d += bias.view(-1, 1)
    # x_conv_out = output_2d.view(batch_size, out_channels, output_height, output_width)

    ## Convolution process
    ## Feel free to use for loop 
    # revise code without using F.unfold
    x_pad = torch.nn.functional.pad(input, (padding, padding, padding, padding), mode='constant', value=0).to(input.device)
    _,_,input_padded_height, input_padded_width = x_pad.shape
    x_unfolded = torch.zeros((input.shape[0], input.shape[1] * kernel_height * kernel_width, output_height * output_width)).to(input.device)
    patch_idx = 0
    for i in range(0, input_padded_height - kernel_height + 1, stride):
        for j in range(0, input_padded_width - kernel_width + 1, stride):
            patch = x_pad[:, :, i:i+kernel_height, j:j+kernel_width]
            x_unfolded[:, :, patch_idx] = patch.reshape(input.shape[0], -1)
            patch_idx += 1
    
    weight_reshaped = weight.view(out_channels, -1)
    output_2d = weight_reshaped @ x_unfolded
    if bias is not None:
        output_2d += bias.view(-1, 1)
    x_conv_out = output_2d.view(batch_size, out_channels, output_height, output_width)

    return x_conv_out
      


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        [hint]
        * The gabor filter kernel has complex number. Be careful.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Paramerter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.b = nn.Parameter(torch.randn(out_channels)) if bias else None
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        if self.bias is not None:
            output = MyFConv2D(x, self.W, self.b, self.stride, self.padding)
        else:
            output = MyFConv2D(x, self.W, stride=self.stride, padding=self.padding)
        return output

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        self.output_height   = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width    = (self.input_width - self.kernel_size) // self.stride + 1
        self.output_channels = self.channel
        self.x_pool_out      = torch.zeros((self.batch_size, self.channel, self.output_height, self.output_width), dtype=x.dtype, device=x.device)

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----
        # revise code without using F.unfold
        x_unfolded =  torch.zeros((x.shape[0], x.shape[1], x.shape[2] // self.stride, x.shape[3] // self.stride, self.kernel_size ** 2)).to(x.device)
        for i in range(0, x.shape[2], self.stride):
            for j in range(0, x.shape[3], self.stride):
                x_unfolded[:,:, i // self.stride, j //self.stride, :] = x[:,:,i:i+self.stride, j:j+self.stride].reshape(x.shape[0],x.shape[1],self.kernel_size**2)

        self.x_pool_out, _ = x_unfolded.max(dim=-1)
        # x_unfolded = F.unfold(x, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride)
        # x_unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        
        return self.x_pool_out


if __name__ == "__main__":

    ## Test your implementation of MyFConv2D as in deliverable 1
    ## You can use the gabor filter kernel from q1.py as the weight and conduct convolution with MyFConv2D
    ## Hint: 
    ## * Be careful with the complex number
    ## * Be careful with the difference between correlation and convolution
    # ----- TODO -----
    pass
