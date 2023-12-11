# foundational optimizer class that supports quantization
# derived from official implementation of the paper: *[Memory Efficient Optimizers with 4-bit States](https://arxiv.org/abs/2309.01507)

import torch

'''
QUANT:
  M:
    ENABLE: True
    BITS: 4
    GROUP_SIZE: 128
    SCALE_TYPE:
      DEFAULT: group
    QUANT_TYPE:
      DEFAULT: nonlinear
    ROUND_TYPE: real-nearest
    Signed: True
    Threshold: 4096
  SQM:
    ENABLE: True
    BITS: 4
    GROUP_SIZE: 128
    SCALE_TYPE:
      DEFAULT: rank1
    QUANT_TYPE:
      DEFAULT: power-1
    ROUND_TYPE: real-nearest
    Signed: False

'''

def create_qmap(quant_type, bit, signed):
    """ create mapping for quantization """
    # default is 4 bit
    # group size 128
    # round type = real-nearest
    # M  = scale via group
    # MQT = nonlinear
    # SQM = scale via rank1
    # SQM_QT = power-1

    if quant_type == 'first_moment':
        # defaults: qt = nonlinear, signed = True, bit =4, 
        return create_dynamic_map(signed, bit-1, bit if signed else bit-1)
    elif quant_type == 'second_moment':
        # defaults = qt = power1, bit = 4, signed = False
        return create_pow_map(bit, signed, 1)

# nonlinear
def create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4):
    """ create dynamic quantization map 
    uses dynamic exponent and fraction. 
    as exponent portion grows, fraction reduces. 
    
    """
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits)-1
    if not signed:
        additional_items *=2  # double range if unsigned
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** (i + non_sign_bits - max_exponent_bits) +1 if signed else 2 ** (i + non_sign_bits- max_exponent_bits +1)+1))

        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10**(-(max_exponent_bits-1) + i)) * means).tolist()

        if signed:
            data+=(-(10** (-(max_exponent_bits-1)+i)) * means).tolist()
        
        if additional_items > 0:
            boundaries = torch.linspace(0.1,1, additional_items +1)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            data += ((10 ** (-(max_exponent_bits-1) +i)) * means).tolist()
            if signed:
                data += (-(10 ** (-(max_exponent_bits-1)+i))* means).tolist()
    data.append(0)
    data.append(1.0)
    data.sort()
    return torch.Tensor(data)

def create_pow_map(bits=4, signed=False, power=1):
    """ create power map 
    for 4bit second moment:
      qmap=tensor([0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000, 0.5625,
        0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 1.0000])
    """
    if not signed:
        qmap = torch.linspace(0,1,(2**bits)+1)[1:] # no zero
        if power > 1:
            qmap = qmap**power
    else:
        qmap = torch.linspace(-1,1,(2**bits)) 
        if power > 1:
            qmap = qmap.sign() * (qmap.abs() ** power)
    return qmap

