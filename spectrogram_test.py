import math

from torch.functional import norm
import torch
import torchaudio
import plotly.express as px


sample_frequency = 400
t = 20  # length in s
t_discrete = torch.arange(0, t * sample_frequency) / sample_frequency
t_discrete = t_discrete.view(4, -1)

signal = torch.zeros(size=t_discrete.size())
signal[0, :] = torch.cos(2 * math.pi * 10 * t_discrete[0, :])
signal[1, :] = torch.cos(2 * math.pi * 25 * t_discrete[1, :])
signal[2, :] = torch.cos(2 * math.pi * 50 * t_discrete[2, :])
signal[3, :] = torch.cos(2 * math.pi * 100 * t_discrete[3, :])

spectrogram_transform = torchaudio.transforms.Spectrogram(
    win_length=10
)

spectrogram = spectrogram_transform(signal.view(-1))

max_value = torch.max(spectrogram)
min_value = torch.min(spectrogram)
normalized = (spectrogram - min_value) / (max_value - min_value)
fig = px.imshow(normalized)
fig.show()

print("debug")
