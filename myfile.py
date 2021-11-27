import torch
from morse_data import MorseData
from morse_model import MorseModel

a = MorseData()
x = a.__getitem__(19)

b = MorseModel()
print(torch.cuda.is_available())
