import pandas as pd
from psrqpy import QueryATNF

# Initialize the query to retrieve pulsar data
query = QueryATNF()

# Get the entire catalog of pulsar data (all pulsars)
pulsars = query.catalogue

# Check how many pulsars are returned
print(f"Number of pulsars in catalog: {len(pulsars)}")

# check the columns and first entries
print(pulsars.columns)
print(pulsars)

# save into csv locally
pulsars.to_csv('complete_ATNF_catalogue.csv', index=False)
