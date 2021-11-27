import torch.nn as nn


from morse_model.train_model import train_model
from morse_model.calc_symbol_accuracy import calc_symbol_accuracy
from morse_model.test_model import test_model
from morse_model.predict import predict
from morse_model.save_model import save_model


class MorseModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Linear(121, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh()
        )

        self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)

        self.classification = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 27),
            nn.LogSoftmax(dim=-2)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x, _ = self.lstm(x)
        x = self.classification(x)
        return x.permute(1, 0, 2)

    def train_model(self, data_loader, n_epochs, optimizer, criterion):
        train_model(self, data_loader, n_epochs, optimizer, criterion)

    def calc_symbol_accuracy(self, batch_x, targets, input_lengths, target_lengths):
        return calc_symbol_accuracy(self, batch_x, targets, input_lengths, target_lengths)

    def test_model(self, data_loader):
        return test_model(self, data_loader)

    def predict(self, input, input_lengths):
        predict(self, input, input_lengths)

    def save_model(self):
        save_model(self)
