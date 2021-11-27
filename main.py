import torch
from torch.utils.data import DataLoader

from morse_data import MorseData, custom_collate_fn
from morse_model import MorseModel


size = 2**14
batch_size = 64
sample_frequency = 4000
carrier_frequency = 500
words_per_minute = 20

train_data = MorseData(size=size,
                       sample_frequency=sample_frequency,
                       carrier_frequency=carrier_frequency,
                       words_per_minute=words_per_minute,)


train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    collate_fn=lambda batch: custom_collate_fn(batch),
    drop_last=True,
    num_workers=4)

morse_model = MorseModel().cuda()
criterion = torch.nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(morse_model.parameters(), lr=0.001)

morse_model.train_model(train_loader, 24, optimizer, criterion)
morse_model.save_model()

test_data = MorseData(size=2**10,
                      sample_frequency=4000,
                      carrier_frequency=500,
                      words_per_minute=20)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    collate_fn=lambda batch: custom_collate_fn(batch),
    num_workers=4)

test_accuracy = morse_model.test_model(test_loader)
print(test_accuracy)

print("debug")
