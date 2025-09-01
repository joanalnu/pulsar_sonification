# video part is based on F0MuteAnimation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy import signal
import tempfile
import os
import subprocess

interval_duration = 60

# Read and clean the data
df = pd.read_csv('complete_ATNF_catalogue.csv')
df_cleaned = df.dropna(subset=['RAJ', 'DECJ', 'F0', 'DATE'], axis=0)


# Function to convert RAJ (HH:MM:SS.sss) to radians
def raj_to_radians(raj_str):
    if '.' not in raj_str:
        raj_str += ".000"
    h, m, s = map(float, raj_str.split(":"))
    decimal_hours = h + m / 60 + s / 3600
    radians = decimal_hours * (2 * np.pi / 24)
    if radians > np.pi:
        radians -= 2 * np.pi
    return radians


# Function to convert DECJ (±DD:MM:SS.sss) to radians
def decj_to_radians(decj_str):
    decj_str = str(decj_str).strip()
    if '-' in decj_str:
        sign = -1
        decj_str = decj_str.replace('-', '').replace('+', '')
    elif '+' in decj_str:
        sign = 1
        decj_str = decj_str.replace('+', '')
    else:
        sign = 1
    if decj_str.count(':') < 2:
        raise ValueError(f"Invalid DECJ format (expected ±DD:MM:SS): {decj_str}")
    parts = decj_str.split(':')
    if '.' not in parts[2]:
        parts[2] += ".000"
    d, m, s = map(float, parts)
    decimal_degrees = d + m / 60 + s / 3600
    if not (-90 <= decimal_degrees <= 90):
        raise ValueError(f"DECJ degrees out of range (-90 to +90): {decimal_degrees}")
    return sign * decimal_degrees * (np.pi / 180)


# Process all data and store with dates and F0 values
pulsar_data = []
for i, row in df_cleaned.iterrows():
    raj, decj = row['RAJ'], row['DECJ']
    try:
        raj_conv = raj_to_radians(raj)
        decj_conv = decj_to_radians(decj)
        pulsar_data.append({
            'ra': raj_conv,
            'dec': decj_conv,
            'f0': row['F0'],
            'date': row['DATE'],
            'original_raj': raj,
            'original_decj': decj
        })
    except Exception as e:
        print(f"Row {i}: Error converting coordinates - RAJ: '{raj}', DECJ: '{decj}' - {e}")
        continue

# Sort by discovery date
pulsar_data.sort(key=lambda x: x['date'])

print(f"Total pulsars to animate: {len(pulsar_data)}")
print(f"Date range: {pulsar_data[0]['date']} to {pulsar_data[-1]['date']}")
print(f"F0 range: {min(x['f0'] for x in pulsar_data):.3f} to {max(x['f0'] for x in pulsar_data):.3f} Hz")

# Normalize frequencies for human hearing range (20Hz-2kHz)
# didn't extend to 20kHz as it sounds way too high
min_freq, max_freq = 20, 2000
f0_values = [x['f0'] for x in pulsar_data]
norm_frequencies = (np.array(f0_values) - min(f0_values)) / (max(f0_values) - min(f0_values))
norm_frequencies = norm_frequencies * (max_freq - min_freq) + min_freq


# Function to generate tone
def generate_tone(frequency, duration=0.3, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate a simple strike sound with exponential decay
    tone = np.sin(2 * np.pi * frequency * t)

    # Add harmonics
    tone += 0.3 * np.sin(2 * np.pi * 2 * frequency * t)
    tone += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)

    # Apply an exponential envelope
    envelope = np.exp(-5 * t)
    tone *= envelope

    # normalize
    tone = tone / np.max(np.abs(tone))

    return tone


# Generate all audio tones
sample_rate = 44100
audio_duration = interval_duration / 1000 # seconds per tone
all_audio = np.array([])

for freq in norm_frequencies:
    tone = generate_tone(freq, audio_duration, sample_rate)
    all_audio = np.concatenate([all_audio, tone])

# Add a little silence at the beginning and end
silence = np.zeros(int(0.5 * sample_rate))
all_audio = np.concatenate([silence, all_audio, silence])

# Save the audio file
audio_file = 'pulsar_audio.wav'
wavfile.write(audio_file, sample_rate, all_audio.astype(np.float32))
print(f"Audio file saved as '{audio_file}'")

# Create figure and axis
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='aitoff')
plt.grid(True)

# Initialize empty scatter plot with F0 for colormap
scatter = ax.scatter([], [], s=40, alpha=0.8, c=[], cmap='viridis',
                     vmin=min(x['f0'] for x in pulsar_data),
                     vmax=max(x['f0'] for x in pulsar_data))
colorbar = plt.colorbar(scatter, ax=ax, label='Rotation Frequency (F0) [Hz]')
title = ax.set_title("Pulsar Discoveries: Year 0", fontsize=14, fontweight='bold')

year_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Store all plotted points
all_ra, all_dec, all_f0, all_dates = [], [], [], []
current_year = 0


def init():
    """Initialize the animation"""
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_array(np.array([]))
    year_text.set_text("")
    return scatter, title, year_text


def update(frame):
    global current_year

    if frame < len(pulsar_data):
        # Add the next pulsar
        pulsar = pulsar_data[frame]
        all_ra.append(pulsar['ra'])
        all_dec.append(pulsar['dec'])
        all_f0.append(pulsar['f0'])
        all_dates.append(pulsar['date'])

        # Update the scatter plot with F0 values
        current_year = pulsar['date']

        scatter.set_offsets(np.c_[all_ra, all_dec])
        scatter.set_array(np.array(all_f0))

        # Update title and year text
        title.set_text(f"Pulsar Discovery Timeline")
        year_text.set_text(f"Year: {current_year}\nTotal: {frame + 1}\nF0: {pulsar['f0']:.3f} Hz")

    return scatter, title, year_text


# Create animation
ani = FuncAnimation(fig, update, frames=range(0, len(pulsar_data), 5),  # Process every 5th frame
                    init_func=init, blit=True, interval=interval_duration, repeat=False)

# Add informational text
info_text = ax.text(0.02, 0.15,
                    f"Total pulsars: {len(pulsar_data)}\nDate range: {pulsar_data[0]['date']} - {pulsar_data[-1]['date']}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()

# First save the video without audio
video_file_no_audio = '5xspeed_pulsar_discovery_video_no_audio.mp4'
ani.save(video_file_no_audio, fps=3, dpi=100, bitrate=1800)
print(f"Video without audio saved as '{video_file_no_audio}'")

# Now combine video and audio using ffmpeg
output_file = '5xspeed_pulsar_discovery_f0_animation_with_audio.mp4'

# Build the ffmpeg command
cmd = [
    'ffmpeg',
    '-y',  # Overwrite output file if it exists
    '-i', video_file_no_audio,  # Input video
    '-i', audio_file,  # Input audio
    '-c:v', 'copy',  # Copy video codec
    '-c:a', 'aac',  # Use AAC audio codec
    '-shortest',  # End when the shortest stream ends
    output_file  # Output file
]

# Run the command
try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Successfully combined video and audio into '{output_file}'")

    # Clean up temporary files
    # os.remove(video_file_no_audio)
    os.remove(audio_file)
    print("Temporary files removed")

except subprocess.CalledProcessError as e:
    print(f"Error combining video and audio: {e}")
    print(f"FFmpeg stderr: {e.stderr}")
except FileNotFoundError:
    print("FFmpeg not found. Please make sure FFmpeg is installed and in your PATH.")