import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv('complete_ATNF_catalogue.csv')

# Clean all required columns at once
df_cleaned = df.dropna(subset=['RAJ', 'DECJ', 'F0', 'DATE'], axis=0)


# Function to convert RAJ (HH:MM:SS.sss) to radians
def raj_to_radians(raj_str):
    # Ensure the string has a proper time format by padding missing decimals to 3 places
    if '.' not in raj_str:
        raj_str += ".000"  # Pad missing decimal seconds with zero
    h, m, s = map(float, raj_str.split(":"))
    decimal_hours = h + m / 60 + s / 3600

    # Convert to radians and adjust to [-π, π] range for Aitoff projection
    radians = decimal_hours * (2 * np.pi / 24)
    if radians > np.pi:
        radians -= 2 * np.pi
    return radians


# Function to convert DECJ (±DD:MM:SS.sss) to radians
def decj_to_radians(decj_str):
    # Convert to string and strip whitespace
    decj_str = str(decj_str).strip()

    # Handle sign - look for +/- anywhere in the string
    if '-' in decj_str:
        sign = -1
        # Remove ALL minus signs and plus signs
        decj_str = decj_str.replace('-', '').replace('+', '')
    elif '+' in decj_str:
        sign = 1
        decj_str = decj_str.replace('+', '')
    else:
        sign = 1  # Default to positive if no sign specified

    # Validate format
    if decj_str.count(':') < 2:
        raise ValueError(f"Invalid DECJ format (expected ±DD:MM:SS): {decj_str}")

    # Pad missing decimal seconds
    parts = decj_str.split(':')
    if '.' not in parts[2]:
        parts[2] += ".000"

    d, m, s = map(float, parts)
    decimal_degrees = d + m / 60 + s / 3600

    # Validate declination range (-90 to +90 degrees)
    if not (-90 <= decimal_degrees <= 90):
        raise ValueError(f"DECJ degrees out of range (-90 to +90): {decimal_degrees}")

    return sign * decimal_degrees * (np.pi / 180)


# Process all data and store with dates and F0 values
pulsar_data = []
for i, row in df_cleaned.iterrows():
    raj, decj = row['RAJ'], row['DECJ']

    try:
        # Try to convert both coordinates
        raj_conv = raj_to_radians(raj)
        decj_conv = decj_to_radians(decj)

        # Store all data together
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

# Add year annotation text
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
    """Update function for each animation frame"""
    global current_year

    if frame < len(pulsar_data):
        # Add the next pulsar
        pulsar = pulsar_data[frame]
        all_ra.append(pulsar['ra'])
        all_dec.append(pulsar['dec'])
        all_f0.append(pulsar['f0'])
        all_dates.append(pulsar['date'])

        # Update current year
        current_year = pulsar['date']

        # Update the scatter plot with F0 values
        scatter.set_offsets(np.c_[all_ra, all_dec])
        scatter.set_array(np.array(all_f0))

        # Update title and year text
        title.set_text(f"Pulsar Discovery Timeline")
        year_text.set_text(f"Year: {current_year}\nTotal: {frame + 1}\nF0: {pulsar['f0']:.3f} Hz")

    return scatter, title, year_text


# Create animation
ani = FuncAnimation(fig, update, frames=len(pulsar_data) + 10,  # +20 for pause at end
                    init_func=init, blit=True, interval=1, repeat=True)

# Add informational text
info_text = ax.text(0.02, 0.15,
                    f"Total pulsars: {len(pulsar_data)}\nDate range: {pulsar_data[0]['date']} - {pulsar_data[-1]['date']}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()

# Save the animation (uncomment to save)
ani.save('pulsar_discovery_f0_animation.mp4', writer='ffmpeg', fps=15, dpi=100, bitrate=1800)
print("Animation saved as 'pulsar_discovery_f0_animation.mp4'")