import torch
import numpy as np
from sklearn import metrics

def metrics_from_raw(raw, y_true):
    '''
    Function to return evaluation metrics for regression
    :param raw: raw prediction obtained from forward pass of model
    :param y_true: actual value
    :return: Mean Absolute Error, Mean Square Error, R2 Score
    '''
    y_pred = raw.cpu().numpy()
    mae = metrics.mean_absolute_error(y_true.numpy(), y_pred)
    mse = metrics.mean_squared_error(y_true.numpy(), y_pred)
    r2_score = metrics.r2_score(y_true.numpy(), y_pred)
    return mae, mse, r2_score

def pred_from_raw(raw):
    pred = torch.argmax(torch.softmax(raw, dim=1), dim=1).cpu().numpy()
    return pred


def metrics_classification(raw, y_true, avg):
    pred = pred_from_raw(raw)
    print(pred)
    f1 = metrics.f1_score(pred, y_true.numpy(), average=avg, zero_division=0)
    prec = metrics.precision_score(pred, y_true.numpy(), average=avg, zero_division=0)
    rec = metrics.recall_score(pred, y_true.numpy(), average=avg, zero_division=0)
    return f1, prec, rec


def accuracy_from_raw(raw, y_true):
    acc =  metrics.accuracy_score(pred_from_raw(raw), y_true.numpy())
    return acc
