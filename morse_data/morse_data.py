""" docstring """

import math
import torch

import torch.nn.utils.rnn as rnn

from torch.utils.data import Dataset

from morse_data.functions import (
    add_noise, generate_morse, modulate, create_spectrogram, normalize)
from morse_data.show_spectrogram import show_spectrogram


class MorseData(Dataset):
    """ docstring """

    def __init__(self,
                 size: int,
                 sample_frequency: int,
                 carrier_frequency: int,
                 words_per_minute: int):

        self.size = size
        self.sample_frequency = sample_frequency
        self.carrier_frequency = carrier_frequency
        self.words_per_minute = words_per_minute

    def __getitem__(self, k):

        length = 8

        # dot/s = (wpm/60) * 50
        # where 50 [dots per word] is derived from PARIS dot number
        samples_per_dot = math.floor(self.sample_frequency / (self.words_per_minute / 60 * 50)
                                     )
        morse_x, morse_y = generate_morse(length, samples_per_dot)

        morse_modulated = modulate(
            morse_x,
            self.sample_frequency,
            self.carrier_frequency)

        morse_x_noised = add_noise(morse_modulated)

        win_length = samples_per_dot
        morse_spectrogram = create_spectrogram(morse_x_noised, win_length)

        input = normalize(morse_spectrogram)
        target = torch.tensor(morse_y)
        # padding is done w/ custom_collate_fn
        # when samples are pulled in a DataLoader
        return input, target

    def __len__(self):
        return self.size

    def show_spectrogram(self):
        """ docstring """
        show_spectrogram(self)


def custom_collate_fn(batch):
    """ docstring """

    batch_x, batch_y = zip(*batch)

    pack_x = rnn.pack_sequence(batch_x, enforce_sorted=False)
    inputs_, input_lengths = rnn.pad_packed_sequence(pack_x, batch_first=True)

    pack_y = rnn.pack_sequence(batch_y, enforce_sorted=False)
    target, target_lengths = rnn.pad_packed_sequence(pack_y, batch_first=True)

    return inputs_, input_lengths, target, target_lengths
