import pandas as pd
import numpy as np

# Setting the random seed for reproducibility
np.random.seed(42)

# Generate dates for the period from Jan 1, 2025 to Dec 31, 2027
date_range = pd.date_range(start='2025-01-01', end='2027-12-31', freq='D')

# Generate wind direction between 0 and 360 degrees
wind_direction = np.random.uniform(0, 360, len(date_range))

# Generate wind speed based on some random fluctuations (say, between 0 and 20 m/s)
wind_speed = np.random.uniform(0, 20, len(date_range))

# Creating the DataFrame
df = pd.DataFrame({
    'Timestamp': date_range,
    'Wind Direction at 100m (deg)': wind_direction,
    'Wind Speed at 100m (m/s)': wind_speed
})

# Display the first few rows of the synthetic dataset
df.head()
