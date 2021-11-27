import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

data = torch.randn(64, 400, 31)

input = rnn.pack_sequence(data, enforce_sorted=False)
input, input_lengths = rnn.pad_packed_sequence(
    input, batch_first=True, total_length=600)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, (4, 1), stride=(2, 1)),
            nn.Conv2d(1, 1, (8, 1), stride=(4, 1))
        )
        self.lstm = nn.LSTM(31, 31, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(31, 7)
        )

    def forward(self, input):
        out1 = self.conv(input.unsqueeze(1))
        print(out1.size())
        out2, _ = self.lstm(out1.squeeze(1))
        out3 = self.linear(out2)
        return out3.repeat_interleave(8, dim=1)


model = Model()
output = model(input)
print(output.size())

criterion = nn.CTCLoss(blank=0)
loss = criterion(output.permute(1, 0, 2),
                 torch.ones((64, 8)),
                 input_lengths,
                 torch.ones(64, dtype=int)
                 )

print("debug")
# # x = torch.randn((64, 100, 1))
# # lstm = nn.LSTM(1, 64, batch_first=True)
# # y, _ = lstm(x)
# # print(y.size())
