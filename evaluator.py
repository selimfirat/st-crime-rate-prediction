import numpy as np

class Evaluator:
    
    def calculate_accuracy(self, y, y_pred):
        
        return 1.0* np.sum(y.flatten() == np.around(y_pred.flatten())) / (33*len(y))
    
    def calculate_macro_f1(self, y, y_pred):
        
        
        f1s= 0.0
        for i in range(33):
            
            topc = (y_pred[:, i] == 1)
            
            top = np.sum(np.logical_and(topc, y[:, i] == 1))
            
            precision = 1.0*top/topc
            recall = 1.0*top/np.sum(y[:, i] == 1)
            
            f1 = 2/(1.0/recall + 1.0/precision)
            
            f1s += f1
    
        return f1s/33