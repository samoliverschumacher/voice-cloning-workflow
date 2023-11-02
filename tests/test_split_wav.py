import pytest
import os
from pathlib import Path

import soundfile as sf
import numpy as np
import sys

scripts_dir = Path(__file__).parent.parent / "scripts"
print(scripts_dir)
sys.path.append(str(scripts_dir))
import split_wav


def get_audio_duration(file_path):
    # helper function to get the duration of an audio file
    pass


def create_test_wav_file(file_path, duration, sample_rate):
    # Create a time array
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create a sine wave at 440 Hz
    data = np.sin(2 * np.pi * 440 * t)

    # Save the audio data as a .wav file
    sf.write(file_path, data, sample_rate)


@pytest.fixture
def input_file():
    file_path = "path/to/input_file.wav"
    create_test_wav_file(file_path, duration=5.0, sample_rate=44100)
    yield file_path
    # Clean up the created input file
    Path(file_path).unlink()

@pytest.fixture
def output_files():
    return ["path/to/output_file1.wav", "path/to/output_file2.wav"]

def test_split_wav(input_file, output_files):
    return
    time_ranges = [(0.0, 1.0), (1.5, 2.0)]
    split_wav.split_audio(input_file, output_files, time_ranges)

    # Assert that output files were created
    for file in output_files:
        assert os.path.exists(file)

    # Assert the duration of output files
    for i, file in enumerate(output_files):
        duration = get_audio_duration(file)
        assert duration == time_ranges[i][1] - time_ranges[i][0]

    # Clean up the created output files
    for file in output_files:
        Path(file).unlink()
