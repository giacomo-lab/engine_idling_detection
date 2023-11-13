# data augmentation

import os
import librosa
import numpy as np
import soundfile as sf

# data augmentation (numpy and librosa)

def time_shift(audio, sr, shift_max, shift_direction):
    shift = np.random.randint(sr * shift_max)
    if shift_direction == 'right':
        shift = -shift
    augmented_audio = np.roll(audio, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_audio[:shift] = 0
    else:
        augmented_audio[shift:] = 0
    return augmented_audio

def add_noise(audio):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + 0.005 * noise
    return augmented_audio

def process_file(file, output_dir):
    # Load audio
    audio, sr = librosa.load(file, sr=None)

    # Apply time shift
    audio_shifted = time_shift(audio, sr, 2, 'right')

    # Apply noise
    audio_noisy = add_noise(audio)

    # Save augmented audio to file, or use it for further processing
    # Define output file path and add "aug" tag to the file name
    filename, file_extension = os.path.splitext(os.path.basename(file))
    output_file_path = os.path.join(output_dir, f"{filename}_aug{file_extension}")

    # Save augmented audio to file
    sf.write(output_file_path, audio_shifted, sr)
    print("saved new file as ", output_file_path)