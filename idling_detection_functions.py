# data augmentation

import os
import numpy as np
import matplotlib.pyplot as plt

import librosa.display
import librosa
import soundfile as sf
from pydub import AudioSegment

# from pydub import AudioSegment
def get_encoding(filename, file_path_dict):
    """ Looks up the file path and prints the encoding type. Doesnt change anything within the file.
        

    Args:
        filename (_string_): name of the file name
        file_path_dict (_dict_): dictionary containing file paths

    Returns:
        _string_: encoding type of each file
    """
    
        
    # Look up the file path
    file_path = file_path_dict.get(filename)
    if file_path is None:
        print(f"File {filename} not found")
        return None

    # Get the encoding details
    try:
        with sf.SoundFile(file_path) as f:
            return f"{f.subtype}"
    except Exception as e:
        print(f"Error with file {filename}: {e}")
        return None


def standardize_audio(audio, target_length, samp_rate, channels):
    """makes all audio files the same length by adding silence at the end of shorter files.
    It also standardizes the sampling rate and the number of channels. Overwrites original files

    Args:
        audio (_path_): path to single audio files
        target_length (_float_): length of longest audio file
        samp_rate (_int_): sampling rate
        channels (_boolean_): mono True or False
    """
    # apply sampling rate of 44100 and mono to all files
    librosa.load(audio, sr=samp_rate, mono = channels) 
    
    # Add silence to shorter files
    current_audio = AudioSegment.from_wav(audio)
    current_length = len(current_audio)
    
    if current_length < target_length: 
        # get difference in duration and calculate the sound of silence
        silence_duration = target_length - current_length
        # create a sound of silence with the needed length   
        silence = AudioSegment.silent(duration=silence_duration)
        #add the sound of silence at the end of the file
        padded_audio = current_audio + silence
        
    else:
        padded_audio = current_audio
    
    padded_audio.export(audio, format='wav')

def time_shift(file_path, output_dir, sr=44100, shift_max=1, shift_direction='right'):
    """Applies a time shift to all audio files. Adds _aug to the end of the output filename.
    
    Args:
        file_path (path): path to the audio files.
        output_dir (_type_): path to output. Same as input is advised.
        sr (int, optional):  Defaults to 44100.
        shift_max (int, optional): Defaults to 1.
        shift_direction (str, optional):  Defaults to 'right'.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Generate a random shift value within the range
    shift = np.random.randint(sr * shift_max)
    if shift_direction == 'right':
        shift = -shift

    augmented_audio = np.roll(audio, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_audio[:shift] = 0
    else:
        augmented_audio[shift:] = 0

    # Define output file path and add "aug" tag to the file name
    filename, file_extension = os.path.splitext(os.path.basename(file_path))
    output_file_path = os.path.join(output_dir, f"{filename}_aug{file_extension}")

    # Save augmented audio to file
    sf.write(output_file_path, augmented_audio, sr)


def create_mel_specs (path_to_wav, path_to_png):
    """Creates mel spec png without axes or legend for each given audio file

    Args:
        path_to_wav (_path_): path to the folder ontaining all wav files
        path_to_png (_path_): destination folder of all png files
    """
    
    #create a list of paths of all wav files
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


    

                