import torch
from torch import nn
from torch import from_numpy

class LSTMMultivariateRegressorPerRegion(nn.Module):
    
    def __init__(self, input_dim, output_dim, lstms = [16, 32, 16]):

        super(LSTMMultivariateRegressorPerRegion, self).__init__()
        
        hdim = lstms[0]
        self.lstms_hdim = lstms
        self.lstms = [
            nn.LSTMCell(input_dim, hdim, bias=True)
        ]
        for lstm_idx in range(1, len(lstms)):
            
            new_hdim = lstms[lstm_idx]
            self.lstms.append(nn.LSTMCell(hdim, new_hdim, bias=True))
            hdim = new_hdim

        self.linear = nn.Linear(hdim, output_dim)
        
        
        self.h_t = [torch.zeros(1, hdim, dtype=torch.float).cuda() for hdim in self.lstms_hdim]
        self.c_t = [torch.zeros(1, hdim, dtype=torch.float).cuda() for hdim in self.lstms_hdim]
        
        
    def forward(self, input):
        preds = []
        
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):

            if input_t.shape[0] == 1:
                input_t = input_t.squeeze(0)

            self.h_t[0], self.c_t[0] = self.lstms[0](input_t, (self.h_t[0], self.c_t[0]))
        
            for i in range(1, len(self.lstms)):
                self.h_t[i], self.c_t[i] = self.lstms[i](self.h_t[i-1], (self.h_t[i], self.c_t[i]))
            
            pred = self.linear(self.h_t[len(self.lstms) - 1])
            preds.append(pred)
        
        preds = torch.stack(preds, 1)
        
        return preds