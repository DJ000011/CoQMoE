import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.build_model import MatMul, GetEx, SoftMax
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, QuantGetex, QuantLayerNorm, QuantSoftmax
from copy import deepcopy


def quant_model(model, input_quant_params={}, weight_quant_params={}, sym=False):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True
    input_quant_params_matmul2['exp_quant'] = True

    # softmax-ex
    quant_params_ex = deepcopy(input_quant_params)
    quant_params_ex['log_quant'] = True
    quant_params_ex['n_bits'] = 4

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                input_quant_params,
                weight_quant_params,
                sym
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'qkv' in name or 'fc1' in name or 'reduction' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, sym)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, sym)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2, sym)
            else:
                new_m = QuantMatMul(input_quant_params, sym)
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, GetEx):
            # GetEx Layer
            idx = idx + 1 if idx != 0 else idx
            if 'getex' in name:
                new_m = QuantGetex(quant_params_ex)
            setattr(father_module, name[idx:], new_m)
            

    return model

def quant_model_minmax(model, input_quant_params={}, weight_quant_params={}, sym=False):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = False
    #input_quant_params_matmul2['exp_quant'] = False

    # softmax-ex
    quant_params_ex = deepcopy(input_quant_params)
    quant_params_ex['log_quant'] = False
    quant_params_ex['n_bits'] = 8

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = False

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                input_quant_params,
                weight_quant_params,
                sym
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            '''if 'qkv' in name or 'fc1' in name or 'reduction' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, sym)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, sym)'''
            new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, sym)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantMatMul(input_quant_params, sym)
            '''if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2, sym)
            else:
                new_m = QuantMatMul(input_quant_params, sym)'''
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, SoftMax):
            # Softmax Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantSoftmax()
            setattr(father_module, name[idx:], new_m)
        '''elif isinstance(m, nn.LayerNorm):
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantLayerNorm(m, input_quant_params)
            setattr(father_module, name[idx:], new_m)'''
        
        '''elif isinstance(m, GetEx):
            # GetEx Layer
            idx = idx + 1 if idx != 0 else idx
            if 'getex' in name:
                new_m = QuantGetex(quant_params_ex)
            setattr(father_module, name[idx:], new_m)'''
            

    return model


def set_quant_state(model, input_quant=False, weight_quant=False, exp_quant=False):
    for name, m in model.named_modules():
        '''if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul, QuantGetex)):
            m.set_quant_state(input_quant, weight_quant)'''
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.set_quant_state(input_quant, weight_quant)
        elif isinstance(m, (QuantMatMul)):
            '''if 'matmul1' in name:
                m.set_quant_state(input_quant, weight_quant)'''
            if 'matmul' in name:
                m.set_quant_state(input_quant, weight_quant)
            elif exp_quant:
                m.set_quant_state(False, weight_quant)
            else:
                m.set_quant_state(True, weight_quant)
        elif isinstance(m, (QuantGetex)):
            m.set_quant_state(exp_quant)
        elif isinstance(m, (QuantLayerNorm)):
            m.set_quant_state(input_quant, weight_quant)
        elif isinstance(m, (QuantSoftmax)):
            m.set_quant_state(input_quant, weight_quant)
            
