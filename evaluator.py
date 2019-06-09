from sklearn.metrics import r2_score
import numpy as np

def score_r2(y_true, y_pred, multioutput="uniform_average"):
    if len(y_true.shape) == 3:
        new_shape = (y_true.shape[0]*y_true.shape[1], y_true.shape[2])
        y_true = y_true.reshape(new_shape)
        y_pred = y_pred.reshape(new_shape)

    return r2_score(y_true, y_pred, multioutput=multioutput)

def score_acc(y_true, y_pred, multioutput="uniform_average"):
    if len(y_true.shape) == 3:
        new_shape = (y_true.shape[0]*y_true.shape[1], y_true.shape[2])
        y_true = y_true.reshape(new_shape)
        y_pred = y_pred.reshape(new_shape)

    y_true[y_true >= 0.5] = 1
    y_true[y_true < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    
    res = np.sum(y_pred == y_true, axis=0) / y_true.shape[0]

    return res.mean()