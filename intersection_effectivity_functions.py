# data augmentation

import os
import librosa
import numpy as np
from matplotlib import pyplot as plt
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
    audio_shifted = time_shift(audio, sr, 1, 'right')

    # Apply noise
    audio_noisy = add_noise(audio)

    # Save augmented audio to file, or use it for further processing
    # Define output file path and add "aug" tag to the file name
    filename, file_extension = os.path.splitext(os.path.basename(file))
    output_file_path = os.path.join(output_dir, f"{filename}_aug{file_extension}")

    # Save augmented audio to file
    sf.write(output_file_path, audio_shifted, sr)
    print("saved new file as ", output_file_path)

def create_mel_specs(path_to_wav, path_to_png):

    #create a list of all wav files needed to be converted: 
    
    file_list = [file for file in os.listdir(path_to_wav)]
    
    for file in file_list:    
        
        #create correct file name for saving
        audio_file = os.path.join(path_to_wav, file)
        output_file = os.path.join(path_to_png, os.path.splitext(file)[0] + '.png')
        
        # load the files and transfrom into np array
        samples, sample_rate = librosa.load(audio_file, sr=44100, mono=True)
        
        #apply a stft transformation    
        stft = librosa.stft(samples)

        #modify it in order to use the mel-scale instead of frenquency
        mel_scale_spec, par = librosa.magphase(stft)
        mel_spec = librosa.feature.melspectrogram(S=mel_scale_spec, sr=sample_rate)     
        
        # Create a figure with the specified size and aspect ratio
        plt.figure(figsize=(2.93, 2.93))  # Adjust the scaling factor based on your preference
        librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max), x_axis='time', y_axis='mel', sr=sample_rate)
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()