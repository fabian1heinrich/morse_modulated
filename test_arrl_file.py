from morse_data import MorseData, custom_collate_fn
from torch.utils.data import DataLoader
import torch
from prepare_file import prepare_file
from morse_model import MorseModel


size = 2**14
batch_size = 64
sample_frequency = 4000
carrier_frequency = 750
words_per_minute = 20

train_data = MorseData(size=size,
                       sample_frequency=sample_frequency,
                       carrier_frequency=carrier_frequency,
                       words_per_minute=words_per_minute)


train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    collate_fn=lambda batch: custom_collate_fn(batch),
    drop_last=True,
    num_workers=4)

train_data.show_spectrogram()

morse_model = MorseModel().cuda()
criterion = torch.nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(morse_model.parameters(), lr=0.001)

# morse_model.train_model(train_loader, 24, optimizer, criterion)

file_string = "210112_20WPM.mp3"
batch_size = 64
sample_length = 25000

input, input_lengths = prepare_file(file_string, batch_size, sample_length)
morse_model.predict(input, input_lengths)

print("debug")
