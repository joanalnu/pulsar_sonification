import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('complete_ATNF_catalogue.csv')

# Clean all required columns at once
df_cleaned = df.dropna(subset=['RAJ', 'DECJ', 'F0', 'DATE'], axis=0)

print(df_cleaned)

# Function to convert RAJ (HH:MM:SS.sss) to radians
def raj_to_radians(raj_str):
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

raj_radians, decj_radians, f0, date = [], [], [], []

for i, row in df_cleaned.iterrows():
    raj, decj = row['RAJ'], row['DECJ']

    try:
        # Try to convert both coordinates
        raj_conv = raj_to_radians(raj)
        decj_conv = decj_to_radians(decj)

        # Only append if both conversions succeed
        raj_radians.append(raj_conv)
        decj_radians.append(decj_conv)
        f0.append(row['F0'])
        date.append(row['DATE'])

    except Exception as e:
        print(f"Row {i}: Error converting coordinates - RAJ: '{raj}', DECJ: '{decj}' - {e}")
        # Skip this row entirely
        continue

# Check if arrays have the same length
print(f"RAJ radians length: {len(raj_radians)}")
print(f"DECJ radians length: {len(decj_radians)}")
print(f"F0 length: {len(f0)}")
print(f"DATE length: {len(date)}")

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='aitoff')
plt.title("Sky Projection of Pulsars")
plt.grid(True)

plt.scatter(raj_radians, decj_radians, s=10, cmap='viridis', c=date)
plt.colorbar(label='Discovery Year')

ax.set_title("Sky Projection of Pulsars")

plt.show()
fig.savefig('pulsar_sky_projection.png', dpi=300)