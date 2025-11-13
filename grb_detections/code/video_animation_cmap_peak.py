import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# read data
names, ra, dec, Dates, peak = [], [], [], [], []
with open('data/BrowseTargets-2204520-1761399506.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line[0] == '|':
            if 'name' not in line:
                empty_beginning, name, ra_single, dec_single, trigger_time, t90, t90_error, t90_start, fluence, fluence_error, flux_1024, flux_1024_error, flux_1024_time, flux_64, flux_64_error, line_jump = line.split('|')
                
                names.append(name.strip(" "))
                ra.append(ra_single.strip(" "))
                dec.append(dec_single.strip(" "))
                Dates.append(trigger_time.strip(" "))
                peak.append(np.float64(flux_1024.strip(" ")))

# functions from github.com/joanalnu/pulsar_sonification
# Function to convert RAJ (HH:MM:SS.sss) to radians
def raj_to_radians(raj_str):
    """
    Converts RA (right ascension) data to radians.
    Input: string with the format: HH:MM:SS.000
    Output: float64
    """
    # Ensure the string has a proper time format by padding missing decimals to 3 places
    if '.' not in raj_str:
        raj_str += ".000"
    h, m, s = map(float, raj_str.split(":"))
    decimal_hours = h + m / 60 + s / 3600

    # Convert to radians and adjust to [-π, π] range for Aitoff projection
    radians = decimal_hours * (2 * np.pi / 24)
    if radians > np.pi:
        radians -= 2 * np.pi
    return radians

# Function to convert DECJ (±DD:MM:SS.sss) to radians
def decj_to_radians(decj_str):
    """
    Converts dec (declination) data to radians.
    Input: string with format: ±DD:MM:SS
    Output: float64
    """
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

    # ValiDate format
    if decj_str.count(':') < 2:
        raise ValueError(f"Invalid DECJ format (expected ±DD:MM:SS): {decj_str}")

    # Pad missing decimal seconds
    parts = decj_str.split(':')
    if '.' not in parts[2]:
        parts[2] += ".000"

    d, m, s = map(float, parts)
    decimal_degrees = d + m / 60 + s / 3600

    # ValiDate declination range (-90 to +90 degrees)
    if not (-90 <= decimal_degrees <= 90):
        raise ValueError(f"DECJ degrees out of range (-90 to +90): {decimal_degrees}")

    return sign * decimal_degrees * (np.pi / 180)

ra_radians = [raj_to_radians(r.replace(' ', ':')) for r in ra]
dec_radians = [decj_to_radians(d.replace(' ', ':')) for d in dec]


# get rid of '-' in Dates (.strip('-') does not work for some reason)
def get_rid_of_minus(ds):
    """
    Deletes symbols other than numbers in a string. I also converts the strings into integers.
    Input: list of strings
    Output: list of int64.
    """
    new_list = []

    for d in ds:
        new_d = ''
        for ch in d:
            if ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                new_d += ch
        new_list.append(np.int64(new_d))
    
    return new_list

Dates = get_rid_of_minus(Dates)

logpeak = np.log10(peak)

print(f'Names {names[:5]}, len: {len(names)}')
print(f'RA {ra[:5]}, len: {len(ra)}')
print(f'DEC {dec[:5]}, len: {len(dec)}')
print(f'RA (rad) {ra_radians[:5]}, len: {len(ra_radians)}')
print(f'DEC (rad) {dec_radians[:5]}, len: {len(dec_radians)}')
print(f'Dates {Dates[:5]}, len: {len(Dates)}')
print(f'Epeak (flux) {peak[:5]}, len: {len(peak)}')



# up until here, everything is the same as in aitoff_projection.py

# create main list for all sub-data-lists for straightforward sorting

cat = pd.DataFrame(  # catalogue data
    {
        'Name':names,
        'RA':ra_radians,
        'DEC':dec_radians,
        'Date':Dates,
        'Epeak':peak,
        'logEpeak':logpeak
    }
)

# sort by Dates
cat.sort_values('Date')

print(f'Total bursts to animate: {len(cat)}')
print(f'Date range: {min(cat['Date'])} to {max(cat['Date'])}')
print(f'Peak flux range: {min(cat['Epeak'])} to {max(cat['Epeak'])}')




###
### Creating the animated video!!!
###


# Create figure and axis
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='aitoff')
plt.grid(True)

# Initialize empty scatter plot with F0 for colormap
scatter = ax.scatter([], [], s=40, alpha=0.8, c=[], cmap='viridis',
                     vmin=min(cat['logEpeak']),
                     vmax=max(cat['logEpeak']))
colorbar = plt.colorbar(scatter, ax=ax, label='LOG Peak flux (`flux_1024`)')
title = ax.set_title("FERMIGBM Burst Detections: Year 0", fontsize=14, fontweight='bold')

# Add year annotation text
year_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Store all plotted points
all_ra, all_dec, all_Dates, all_peaks = [], [], [], []
current_year = 0


def init():
    """Initialize the animation"""
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_array(np.array([]))
    year_text.set_text("")
    return scatter, title, year_text


def Update(frame):
    """Update function for each animation frame"""
    global current_year

    if frame < len(cat):
        # Add the next burst
        burst = cat.iloc[frame]
        all_ra.append(burst['RA'])
        all_dec.append(burst['DEC'])
        all_Dates.append(burst['Date'])
        all_peaks.append(burst['logEpeak'])

        # Update current year
        current_year = burst['Date']

        # Update the scatter plot with Epeak values
        scatter.set_offsets(np.c_[all_ra, all_dec])
        scatter.set_array(np.array(all_peaks))

        # Update title and year text
        title.set_text(f"FERMIGBM Burst Detection Timeline")
        year_text.set_text(f"Year: {current_year}\nTotal: {frame + 1}\nPeakFlux: {burst['Epeak']:.3f} Hz")

    return scatter, title, year_text


# Create animation
ani = FuncAnimation(fig, Update, frames=len(cat) + 10,  # +10 for pause at end
                    init_func=init, blit=True, interval=1, repeat=True)

# Add informational text
info_text = ax.text(0.02, 0.15,
                    f"Total bursts: {len(cat)}\nDate range: {min(cat['Date'])} - {max(cat['Date'])}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()

# Save the animation (uncomment to save)
ani.save('figures/fermigbm_burst_catalogue_animation_cmap_flux1024.mp4', writer='ffmpeg', fps=15, dpi=100, bitrate=1800)