import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter
from copy import deepcopy

from .quantizer import UniformQuantizer, SymmetricQuantizer, LogSqrt2Quantizer


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    """
    def __init__(self,   
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                input_quant_params={},
                weight_quant_params={},
                sym=False):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        input_quant_params_conv = deepcopy(input_quant_params)
        input_quant_params_conv['n_bits'] = 8
        if sym is True:
            self.input_quantizer = SymmetricQuantizer(**input_quant_params_conv)
            self.weight_quantizer = SymmetricQuantizer(**weight_quant_params)
        else:
            self.input_quantizer = UniformQuantizer(**input_quant_params_conv)
            self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x):
        """
        using quantized weights to forward input x
        """
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

        return out


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 sym=False):
        super(QuantLinear, self).__init__(in_features, out_features)
        if sym is True:
            self.input_quantizer = SymmetricQuantizer(**input_quant_params)
            self.weight_quantizer = SymmetricQuantizer(**weight_quant_params)
        else:   
            self.input_quantizer = UniformQuantizer(**input_quant_params)
            self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x):
        """
        using quantized weights to forward input x
        """

        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.linear(x, weight=w, bias=self.bias)

        return out
        

class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={},
                 sym=False):
        super(QuantMatMul, self).__init__()

        input_quant_params_matmul = deepcopy(input_quant_params)
        if 'exp_quant' not in input_quant_params_matmul:
            if 'log_quant' in input_quant_params_matmul and input_quant_params_matmul['log_quant'] is True:
                input_quant_params_matmul.pop('log_quant')
                self.quantizer_A = LogSqrt2Quantizer(**input_quant_params_matmul)
            else:
                if sym is True:
                    self.quantizer_A = SymmetricQuantizer(**input_quant_params_matmul)
                else:
                    self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        else:
            input_quant_params_matmul.pop('exp_quant')
            input_quant_params_matmul.pop('log_quant')

        if sym is True:
            self.quantizer_B = SymmetricQuantizer(**input_quant_params_matmul)
        else:
            self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)

        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, A, B):
        if self.use_input_quant:
            A = self.quantizer_A(A)
        if self.use_weight_quant:
            B = self.quantizer_B(B)
        
        out = A @ B
        return out
class QuantLayerNorm(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 norm,
                 input_quant_params={},):
        super(QuantLayerNorm, self).__init__()
        self.norm = norm
        self.input_quantizer = SymmetricQuantizer(**input_quant_params)

        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantLayerNorm, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, x):
        if self.use_input_quant:
            x = self.input_quantizer(x)

        out = self.norm(x)

        return out

class QuantSoftmax(nn.Module):
    def __init__(self, input_quant_params={}):
        super(QuantSoftmax, self).__init__()
        input_quant_params_softmax = deepcopy(input_quant_params)
        self.quantizer_softmax = UniformQuantizer(**input_quant_params_softmax)
        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantSoftmax, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
    def forward(self, x):
        if self.use_input_quant:
            x = self.quantizer_softmax(x)
        # max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        # e_x = torch.exp(x - max_vals)
        # sum_e_x = torch.sum(e_x, dim=-1, keepdim=True)
        # out = e_x / sum_e_x
        out = F.softmax(x, dim=-1)
        return out
class QuantGetex(nn.Module):
    def __init__(self,
                 input_quant_params={}):
        super(QuantGetex, self).__init__()
        input_quant_params_getex = deepcopy(input_quant_params)
        if 'log_quant' in input_quant_params_getex:
            if input_quant_params_getex['log_quant'] is True:
                self.quantizer_ex = LogSqrt2Quantizer(**input_quant_params_getex)
                input_quant_params_getex.pop('log_quant')
            else:
                input_quant_params_getex.pop('log_quant')
                self.quantizer_ex = UniformQuantizer(**input_quant_params_getex)
            self.use_input_quant = True

    def __repr__(self):
        s = super(QuantGetex, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
    def forward(self, x):
        max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        out = torch.exp(x - max_vals)
        if self.use_input_quant:
            out = self.quantizer_ex(out)
        return out