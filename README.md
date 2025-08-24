pulsar_sonification
---

This repository contains code I used to sonify pulsar data. The aim of this project was to create an audio product that represents the discovery of pulsars over time since 1978, when the first pulsar was discovered, until 2025. The data was fetched from the ATNF Pulsar Catalogue using the `psrqpy` library.

In order to sonify the pulsar data, I mapped the pulsar spin frequency (F0) into audible frequences using a logarithmic scale to ensure that the wide range of pulsar frequencies could be represented within the human hearing range (20 Hz to 5 kHz).

## Repository Structure

The structure of the files is the following, in chronological order of creation:
- `ATNFfetching.py`: Fetches all pulsar data from the ATNF Pulsar Catalogue (complete catalogue between 1978 and 2025) $\to$ `complete_ATNF_catalogue.csv`.
- `aitoffProjection.py`: Generates `png` figure of pulsar positions using Aitoff projection.
- `muteAnimation.py`: Generates `mp4` animation of pulsar positions using Aitoff projection displaying them chronologically by date recorded in ATNF data $\to$ `muteAnimation.mp4`.
- `F0MuteAnimation.py`: Similar to `muteAnimation.py` but colormap is created following a frequency instead of dates $\to$ `F0MuteAnimation.mp4`.
- `soundAnimation.py`: Same of `F0MuteAnimation.py`, now with sound $\to$ `soundAnimation.mp4`.


## Requirements
- Python 3.x (this code was developed using Python 3.12)
- pandas 
- numpy 
- matplotlib 
- psrqpy 
- scipy