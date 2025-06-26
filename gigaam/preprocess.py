from subprocess import CalledProcessError, run
from typing import Tuple

import torch
import torchaudio
from torch import Tensor, nn

from scipy.io import wavfile
import numpy as np
import subprocess
import tempfile
import os
import threading

SAMPLE_RATE = 16000


def get_audio_duration(input_file):
    """Gets the duration of an audio file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_file,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout)
    except Exception as e:
        print(f"Error getting duration with ffprobe: {e}")
        return None


def load_audio(input_file, sr=SAMPLE_RATE) -> torch.Tensor:
    """
        A more efficient implementation of loading and converting given audio into 
        a required tensor. Uses multithreading to convert long audios to wav more efficiently
        by splitting the audio based on its length using `ffmpeg` and then concatenating them 
        into a single array.
    """
    try:
        sr_actual, x = wavfile.read(input_file)
        if (sr_actual != sr) or (len(x.shape) != 1):
            raise Exception("Not a 16kHz wav mono channel file!")

        audio_signal = None

        if x.dtype == np.int16:
            audio_signal = x.astype(np.float32) / 32768.0
        elif x.dtype != np.float32:
            audio_signal = x.astype(np.float32)
        num_threads = 1

    except:
        total_duration = get_audio_duration(input_file)

        if total_duration is None:
            raise RuntimeError("Could not determine audio duration.")

        # A totally scientific approach, btw xD
        if total_duration < 60:
            num_threads = 1
        elif total_duration < 300:
            num_threads = 2
        elif total_duration < 900:
            num_threads = 4
        else:
            num_threads = 8

        chunk_duration = total_duration / num_threads
        audio_chunks = [None] * num_threads  # Initialize list with None placeholders
        threads = []

        with tempfile.TemporaryDirectory() as tmpdir:

            def process_chunk(i, start_time, end_time):
                wav_file = f"{tmpdir}/tmp_{i}.wav"
                ret_code = os.system(
                    f'ffmpeg -hide_banner -loglevel panic -hwaccel auto -ss {start_time} -to {end_time} -i "{input_file}" -threads 1 -acodec pcm_s16le -ac 1 -ar {sr} "{wav_file}" -y'
                )
                if ret_code != 0:
                    raise RuntimeError(
                        "ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!"
                    )
                _, x = wavfile.read(wav_file)

                if x.dtype == np.int16:
                    audio_chunks[i] = x.astype(np.float32) / 32768.0
                elif x.dtype != np.float32:
                    audio_chunks[i] = x.astype(np.float32)

            for i in range(num_threads):
                start_time = i * chunk_duration
                end_time = (
                    (i + 1) * chunk_duration if i < num_threads - 1 else total_duration
                )
                thread = threading.Thread(
                    target=process_chunk, args=(i, start_time, end_time)
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            audio_signal = np.concatenate(audio_chunks)

    return torch.tensor(audio_signal)


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(self, sample_rate: int, features: int):
        super().__init__()
        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
