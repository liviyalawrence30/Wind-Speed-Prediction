import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Step 1: Load and rename columns
df = pd.read_csv(r"C:\miniproject\wind_data_2009.csv")
df.columns = ['wind speed at 100m (m/s)', 'wind direction at 100m (deg)', 'timestamp']

# Step 2: Save cleaned data for future use
df.to_csv("inputs.csv", index=False)

# Step 3: Fit MinMaxScaler and save it
scaler = MinMaxScaler()
scaler.fit(df[['wind direction at 100m (deg)', 'timestamp']])
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 4: Load scaler and model
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

model = load_model("lstm_wind_model_cgo.keras")

# Step 5: Scale input features
X = loaded_scaler.transform(df[['wind direction at 100m (deg)', 'timestamp']])

# Step 6: Reshape for LSTM [samples, time_steps, features]
X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Step 7: Predict
predictions = model.predict(X_reshaped)

# Step 8: Inverse scale predictions
# Create a dummy array with predictions at index 0 (wind speed), original inputs at index 1 and 2
dummy_full = np.concatenate([predictions, X], axis=1)
inv_scaled = scaler.inverse_transform(dummy_full)

# Extract only wind speed
predicted_wind_speed = inv_scaled[:, 0]

# Step 9: Add predictions and extract year
df['Predicted Wind Speed at 100m (m/s)'] = predicted_wind_speed
df['Year'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).year)

# Step 10: Calculate MAE
mae = mean_absolute_error(df['wind speed at 100m (m/s)'], df['Predicted Wind Speed at 100m (m/s)'])
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Step 11: Save predictions to CSV
df[['wind direction at 100m (deg)', 'timestamp', 'Predicted Wind Speed at 100m (m/s)', 'Year']].to_csv('predictions.csv', index=False)

# Step 12: Plot predictions grouped by year
plt.figure(figsize=(12, 6))
df.groupby('Year')['Predicted Wind Speed at 100m (m/s)'].mean().plot(marker='o')
plt.title('Average Predicted Wind Speed per Year')
plt.xlabel('Year')
plt.ylabel('Wind Speed at 100m (m/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig("wind_predictions_by_year.png")
plt.show()
