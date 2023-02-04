import torch

def torch2mi(x):
    """x: Bx3"""
    ret = x[:,[0,2,1]]
    ret = torch.tensor([[1,1,-1]],device=x.device)*ret
    return ret

def mi2torch(x):
    """x:Bxe"""
    ret = torch.tensor([[1,1,-1]],device=x.device)*x
    return ret[:,[0,2,1]]
