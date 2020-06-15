import torch
import pandas as pd
import numpy as np
from sklearn import metrics

def pred_from_raw(raw):
    pred = torch.argmax(torch.softmax(raw, dim=1), dim=1).numpy()
    return pred

def f1_from_raw(raw, y_true, avg, classes):
    pred = pred_from_raw(raw)
    f1 = metrics.f1_score(pred,
                            y_true.numpy(),
                            average=avg,
                            labels=classes,
                            zero_division=0)
    prec = metrics.precision_score(pred,
                                   y_true.numpy(),
                                   average=avg,
                                   labels=classes,
                                   zero_division=0)
    rec = metrics.recall_score(pred,
                                y_true.numpy(),
                                average=avg,
                                labels=classes,
                                zero_division=0)
    return f1, prec, rec

def accuracy_from_raw(raw, y_true):
    acc =  metrics.accuracy_score(pred_from_raw(raw), y_true.numpy())
    return acc
