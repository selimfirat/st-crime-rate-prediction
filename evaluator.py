from sklearn.metrics import r2_score
    
def score_r2(y_true, y_pred, multioutput="uniform_average"):
    if len(y_true.shape) == 3:
        new_shape = (y_true.shape[0]*y_true.shape[1], y_true.shape[2])
        y_true = y_true.reshape(new_shape)
        y_pred = y_pred.reshape(new_shape)

    return r2_score(y_true, y_pred, multioutput=multioutput)
