import numpy as np
import matplotlib.pyplot as plt

# read data
names, ra, dec, dates, peak = [], [], [], [], []
with open('data/BrowseTargets-2204520-1761399506.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line[0] == '|':
            if 'name' not in line:
                empty_beginning, name, ra_single, dec_single, trigger_time, t90, t90_error, t90_start, fluence, fluence_error, flux_1024, flux_1024_error, flux_1024_time, flux_64, flux_64_error, line_jump = line.split('|')
                
                names.append(name.strip(" "))
                ra.append(ra_single.strip(" "))
                dec.append(dec_single.strip(" "))
                dates.append(trigger_time.strip(" "))
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


ra_radians = [raj_to_radians(r.replace(' ', ':')) for r in ra]
dec_radians = [decj_to_radians(d.replace(' ', ':')) for d in dec]


# get rid of '-' in dates (.strip('-') does not work for some reason)
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


dates = get_rid_of_minus(dates)

print(f'Names {names[:5]}, len: {len(names)}')
print(f'RA {ra[:5]}, len: {len(ra)}')
print(f'DEC {dec[:5]}, len: {len(dec)}')
print(f'RA (rad) {ra_radians[:5]}, len: {len(ra_radians)}')
print(f'DEC (rad) {dec_radians[:5]}, len: {len(dec_radians)}')
print(f'Dates {dates[:5]}, len: {len(dates)}')
print(f'Epeak (flux) {peak[:5]}, len: {len(peak)}')

# Create figure!
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='aitoff')
plt.title("Sky Projection of FERMIGBM Burst Catalogue")
plt.grid(True)

plt.scatter(ra_radians, dec_radians, s=10, cmap='viridis', c=dates)
plt.colorbar(label='Discovery Date')

ax.set_title("Sky Projection of FERMIGBM Burst Catalogue")

plt.show()
fig.savefig('figures/static_aitoff_projection.png', dpi=600)