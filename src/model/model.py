# 모델 정의 및 학습 코드
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YourModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        output, _ = self.rnn(x, (h0, c0))
        output = self.fc(output[-1, :, :])
        return output
