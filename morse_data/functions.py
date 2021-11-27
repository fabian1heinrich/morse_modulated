import random
import math
import torch
import torchaudio

from morse_data.morse_dict import morse_dict


def generate_morse(length, samples_per_dot):

    morse_seq = random.choices(list(morse_dict.items()), k=length)
    morse_seq = list(zip(*morse_seq))

    morse_y = morse_seq[0]
    morse_x = []

    morse_x.extend([0.0, 0.0, 0.0]
                   * dilation(samples_per_dot))
    for letter in morse_seq[1]:
        for item in letter:
            morse_x.extend(
                [item] * dilation(samples_per_dot))

        morse_x.extend(
            [0.0, 0.0, 0.0] * dilation(samples_per_dot))

    return morse_x, [y + 1 for y in morse_y]


def modulate(m_signal, sample_frequency, carrier_frequency):

    m_signal = torch.tensor(m_signal)
    time_discrete = torch.arange(0, len(m_signal)) / sample_frequency
    carrier_wave = torch.sin(2 * math.pi * carrier_frequency * time_discrete)

    y_signal = m_signal * carrier_wave

    return y_signal


def add_noise(signal):

    noise = torch.normal(mean=0.0, std=1/4, size=signal.size())

    return signal + noise


def create_spectrogram(y_signal, win_length):

    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=win_length,
        win_length=win_length,
        hop_length=win_length
    )

    spectrogram = spectrogram_transform(y_signal)
    return spectrogram.transpose_(0, 1)


def normalize(spectrogram):

    min_value = torch.min(spectrogram)
    max_value = torch.max(spectrogram)
    normalized_spectrogram = (spectrogram - min_value) / \
        (max_value - min_value)

    return normalized_spectrogram


def dilation(sample_length_dot: int):

    dil_factor = max(min(random.normalvariate(1.0, 1/6), 1.5), 0.5)
    dil = math.floor(dil_factor * sample_length_dot)
    return dil
