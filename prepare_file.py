import torch
import torchaudio
import plotly.express as px

from morse_data import create_spectrogram, normalize


def prepare_file(file_string, batch_size, sample_length):

    data, sample_rate = torchaudio.load(file_string)
    # sample rate = 8000 --> downsample by factor 2
    data = data[0, 0::2]
    n_samples = len(data) // sample_length
    signal = data[0:n_samples * sample_length].reshape(-1, sample_length)

    batch = torch.zeros((batch_size, 105, 121))
    offset = 5
    for i in range(batch_size):
        spectrogram = create_spectrogram(
            signal[offset + i, :], win_length=240)
        normalized_spectrogram = normalize(spectrogram)
        batch[i, :, :] = normalized_spectrogram

    input_lengths = batch.size()[1] * torch.ones(batch_size, dtype=int)
    return batch, input_lengths
