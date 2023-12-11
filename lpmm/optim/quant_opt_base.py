# foundational optimizer class that supports quantization
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

    if quant_type == 'nonlinear':
        # defaults: qt = nonlinear, signed = True, bit =4, 
        return create_dynamic_map(signed, bit-1, bit if signed else bit-1)
    elif quant_type == 'power-1':
        # defaults = qt = power1, bit = 4, signed = False
        return create_pow_map(bit, signed, 1)

