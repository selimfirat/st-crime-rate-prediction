import torch
import numpy as np
from tensorboardX import SummaryWriter

def init_seeds():
    torch.manual_seed(0)
    np.random.seed(0)


def init_cuda(device_id=6):
    torch.cuda.set_device(6)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

writers = []

def get_logger(model_name):
    
    writer = SummaryWriter("logs/" + model_name)
    
    writers.append(writer)
    
    return writer

def clear_caches():
    torch.cuda.empty_cache()
    global writers
    
    for writer in writers:
        writer.close()
    
    del writers