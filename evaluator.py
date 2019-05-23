import numpy as np

class Evaluator:
    
    def score_r2(self, y_true, y_pred, multioutput="uniform_average"):
        new_shape = (y_true.shape[0]*y_true.shape[1], y_true.shape[2])
        y_true = y_true.reshape(new_shape)
        y_pred = y_pred.reshape(new_shape)

        return r2_score(y_true, y_pred, multioutput=multioutput)
