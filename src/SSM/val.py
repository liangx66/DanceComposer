import torch
import numpy as np

def compute_accuracy(out, tgt):
    # Empty
    if (len(tgt) == 0):
        return torch.tensor(1.0)
    num_right = (out == tgt)
    num_right = torch.sum(num_right).float()
    epsilon = 1e-8
    acc = num_right / (len(tgt)+epsilon)
    return acc

def compute_f1_score(out, tgt):
    tp = (tgt * out).sum().to(torch.float32)
    tn = ((1 - tgt) * (1 - out)).sum().to(torch.float32)
    fp = ((1 - tgt) * out).sum().to(torch.float32)
    fn = (tgt * (1 - out)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon) 
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1