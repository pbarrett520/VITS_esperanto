import wave
import numpy as np

# Function to load WAV file and return numpy array of frames, sample rate, and duration
def load_wav_as_array(file_path):
    with wave.open(file_path, 'r') as wav_file:
        n_frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration = n_frames / sample_rate
        audio_data = wav_file.readframes(n_frames)
        # Convert audio bytes to a numpy array based on the sample width
        if wav_file.getsampwidth() == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif wav_file.getsampwidth() == 4:
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError("Unsupported sample width")
        return (audio_array, sample_rate, duration)

# Function to calculate the speech rate (words per minute)
def calculate_speech_rate(duration, total_words=16):
    return total_words / duration * 60

# Function to find significant pauses in the audio, which might indicate punctuation
def find_significant_pauses(audio, sample_rate, threshold=0.02):
    audio_normalized = audio / np.max(np.abs(audio))
    pauses = np.abs(audio_normalized) < threshold
    pause_lengths = []
    current_pause_length = 0
    for is_pause in pauses:
        if is_pause:
            current_pause_length += 1
        elif current_pause_length > 0:
            pause_lengths.append(current_pause_length / sample_rate)
            current_pause_length = 0
    significant_pauses = [pause for pause in pause_lengths if pause > 0.2]
    return significant_pauses

# File paths
file_path_1 = 'esperanto_final.wav'
file_path_2 = 'esperanto_no_punct.wav'

# Load WAV files and calculate durations
audio_array_1, sample_rate_1, duration_1 = load_wav_as_array(file_path_1)
audio_array_2, sample_rate_2, duration_2 = load_wav_as_array(file_path_2)

# Calculate speech rates using the actual durations
speech_rate_1 = calculate_speech_rate(duration_1)
speech_rate_2 = calculate_speech_rate(duration_2)

# Find significant pauses
significant_pauses_1 = find_significant_pauses(audio_array_1, sample_rate_1)
significant_pauses_2 = find_significant_pauses(audio_array_2, sample_rate_2)

print("Speech Rate 1:", speech_rate_1)
print("Speech Rate 2:", speech_rate_2)
print("Significant Pauses 1:", significant_pauses_1)
print("Significant Pauses 2:", significant_pauses_2)
