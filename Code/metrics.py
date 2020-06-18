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

