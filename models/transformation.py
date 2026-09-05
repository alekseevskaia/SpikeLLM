
import torch
import pdb

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)



def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts
        else:
            fc.temp_bias = fc.weight@shifts  # for Qwen3 the shape is (1024,)
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    fc1_out = fc1.out_features #4096
    fc2_in = fc2.in_features #4096 

    if fc1_out < fc2_in:
        num_groups = fc2_in // fc1_out 
        scales_fc1 = scales.reshape(-1, num_groups).mean(dim=1)  
        if shifts is not None:
            shifts_fc1 = shifts.reshape(-1, num_groups).mean(dim=1)
        else:
            shifts_fc1 = None
    else:
        scales_fc1 = scales
        shifts_fc1 = shifts
    
    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - shifts_fc1
        fc1.temp_bias = fc1.temp_bias/scales_fc1.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales_fc1.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/scales_fc1.view(-1)
        fc1.temp_weight = fc1.weight/scales_fc1.view(-1,1)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    q_out = q_proj.out_features
    k_out = k_proj.out_features

    if q_out > k_out:
        num_groups = q_out // k_out
        scales_q = torch.repeat_interleave(scales, repeats=num_groups)
        scales_k = scales
    else:
        scales_q = scales
        scales_k = scales

    q_proj.temp_weight = q_proj.temp_weight / scales_q.view(-1, 1)
    q_proj.temp_bias = q_proj.temp_bias / scales_q.view(-1)
    k_proj.temp_weight = k_proj.temp_weight * scales_k.view(-1, 1)
    k_proj.temp_bias = k_proj.temp_bias * scales_k.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))


def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    fc1_out = fc1.out_features
    fc2_in = fc2.in_features

    if fc1_out < fc2_in:
        num_groups = fc2_in // fc1_out
        scales_fc1 = scales.reshape(-1, num_groups).mean(dim=1)
        if shifts is not None:
            shifts_fc1 = shifts.reshape(-1, num_groups).mean(dim=1)
        else:
            shifts_fc1 = None
    else:
        scales_fc1 = scales
        shifts_fc1 = shifts

    fc1.bias.sub_(shifts_fc1)
    fc1.bias.div_(scales_fc1.view(-1))
    fc1.weight.div_(scales_fc1.view(-1, 1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales,):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    q_out = q_proj.out_features
    k_out = k_proj.out_features

    if q_out > k_out:
        num_groups = q_out // k_out
        scales_q = torch.repeat_interleave(scales, repeats=num_groups)
        scales_k = scales
    else:
        scales_q = scales
        scales_k = scales

    q_proj.weight.div_(scales_q.view(-1, 1))
    q_proj.bias.div_(scales_q.view(-1))
    k_proj.weight.mul_(scales_k.view(-1, 1))
    k_proj.bias.mul_(scales_k.view(-1))